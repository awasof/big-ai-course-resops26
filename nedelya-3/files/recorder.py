# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "sounddevice; sys_platform != 'darwin'",
#   "pyobjc-framework-ScreenCaptureKit; sys_platform == 'darwin'",
#   "pyobjc-framework-CoreMedia; sys_platform == 'darwin'",
#   "pyobjc-framework-AVFoundation; sys_platform == 'darwin'",
#   "pyobjc-framework-libdispatch; sys_platform == 'darwin'",
# ]
# ///
"""Запись звука в WAV: микрофон, звук из компьютера (то, что говорит собеседник в звонке) или оба сразу.

Результат — моно, 16 кГц, 16 бит: Whisper всё равно переводит звук в такой вид, а файл получается маленьким
(около 2 МБ в минуту). Файл пишется по ходу дела и каждые две секунды получает правильный заголовок,
поэтому запись не пропадает, даже если программа упадёт.

Звук из компьютера на Mac берётся через ScreenCaptureKit: это тот же системный механизм, что у записи экрана,
поэтому macOS просит разрешение «Запись экрана и системного звука». Картинка с экрана не сохраняется:
запрашивается только звук. Драйверы ставить не нужно. Микрофон на Mac берётся через AVFoundation (то же,
что использует система), а не через PortAudio: у PortAudio на Mac остановка записи иногда зависала насмерть.
На Windows пока работает только микрофон (через sounddevice).

Проверка без окна:  uv run recorder.py mic|system|both [секунды] [файл.wav]
"""

import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RATE = 16000  # частота записи; Whisper работает с ней же
LAG = 0.4  # запись отстаёт от часов на 0,4 с: так в потоке не возникает дырок из-за неровной подачи звука
DEFAULT_FOLDER = Path.home() / "Documents" / "Transcribe"
CAN_RECORD_SYSTEM = sys.platform == "darwin"


class RecorderError(Exception):
    """Ошибка, текст которой уже написан для человека."""


def to_rate(mono: np.ndarray, rate: float) -> np.ndarray:
    """Приводит одноканальный звук к 16 кГц. Целое число раз (48000, 32000) — усреднением, иначе линейной подстановкой."""
    if rate == RATE:
        return mono
    if rate % RATE == 0:
        step = int(rate // RATE)
        usable = len(mono) // step * step
        return mono[:usable].reshape(-1, step).mean(axis=1)
    count = int(len(mono) * RATE / rate)
    return np.interp(np.linspace(0, len(mono) - 1, count), np.arange(len(mono)), mono) if count > 1 else mono[:0]


class WavWriter:
    """WAV, который пишется по частям. Заголовок обновляется раз в две секунды и при закрытии."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = open(path, "wb")
        self.samples = 0
        self.patched = time.monotonic()
        self._patch()

    def _patch(self) -> None:
        data = self.samples * 2
        header = (
            b"RIFF" + struct.pack("<I", 36 + data) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16)
            + b"data" + struct.pack("<I", data)
        )
        self.file.seek(0)
        self.file.write(header)
        self.file.seek(0, 2)
        self.file.flush()
        self.patched = time.monotonic()

    def write(self, mono: np.ndarray) -> None:
        pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2")
        self.file.write(pcm.tobytes())
        self.samples += len(pcm)
        if time.monotonic() - self.patched > 2:
            self._patch()

    def close(self) -> None:
        self._patch()
        self.file.close()


class Source:
    """Источник звука: складывает присланные куски в очередь, из которой запись забирает нужное количество."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.parts: list[np.ndarray] = []
        self.count = 0
        self.peak = 0.0  # самый громкий момент с прошлого опроса level()
        self.silent_since: float | None = None  # с какого момента идёт «цифровая тишина» (ровно ноль)
        self.error: str | None = None
        self.received = 0  # сколько всего отсчётов пришло

    def push(self, mono: np.ndarray) -> None:
        if not len(mono):
            return
        top = float(np.abs(mono).max())
        with self.lock:
            self.parts.append(mono.astype(np.float32, copy=False))
            self.count += len(mono)
            self.received += len(mono)
            self.peak = max(self.peak, top)
            if top > 0:
                self.silent_since = None
            elif self.silent_since is None:
                self.silent_since = time.monotonic()

    def take(self, wanted: int | None) -> np.ndarray:
        """Забирает до wanted отсчётов с начала очереди (все, если wanted не задано)."""
        with self.lock:
            if not self.parts:
                return np.zeros(0, np.float32)
            joined = np.concatenate(self.parts) if len(self.parts) > 1 else self.parts[0]
            if wanted is None or wanted >= len(joined):
                self.parts, self.count = [], 0
                return joined
            self.parts, self.count = [joined[wanted:]], len(joined) - wanted
            head = joined[:wanted]
            # Часы источника чуть быстрее часов компьютера — очередь понемногу растёт: лишнее старше секунды выбрасываем.
            if self.count > RATE:
                self.parts, self.count = [self.parts[0][-RATE // 3:]], RATE // 3
            return head

    def level(self) -> float:
        with self.lock:
            peak, self.peak = self.peak, 0.0
        return peak

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


def buffer_to_mono(cm, buffer) -> np.ndarray:
    """Звук из буфера macOS (CMSampleBuffer) в моно 16 кГц. Понимает и «подряд по каналам», и «вперемешку»."""
    description = cm.CMAudioFormatDescriptionGetStreamBasicDescription(cm.CMSampleBufferGetFormatDescription(buffer))
    rate, flags, channels, bits = description[0], description[2], description[6], description[7]
    frames = cm.CMSampleBufferGetNumSamples(buffer)
    block = cm.CMSampleBufferGetDataBuffer(buffer)
    _status, raw = cm.CMBlockBufferCopyDataBytes(block, 0, cm.CMBlockBufferGetDataLength(block), None)
    data = np.frombuffer(raw, np.float32 if bits == 32 else np.int16)
    if bits != 32:
        data = data.astype(np.float32) / 32768
    if flags & 32:  # каналы идут подряд, один за другим
        mono = data[: frames * channels].reshape(channels, frames).mean(axis=0)
    else:  # каналы перемешаны отсчёт за отсчётом
        mono = data[: frames * channels].reshape(frames, channels).mean(axis=1)
    return to_rate(mono, rate)


_MIC_CLASS = None


def _mic_class():
    """Класс-посредник для микрофона. Как и для звука компьютера, создаётся один раз."""
    global _MIC_CLASS
    if _MIC_CLASS is None:
        import objc
        from Foundation import NSObject

        class MicTap(NSObject):
            def initWithOwner_(self, owner):
                self = objc.super(MicTap, self).init()
                if self is not None:
                    self.owner = owner
                return self

            def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, buffer, connection):
                self.owner.on_buffer(buffer)

        _MIC_CLASS = MicTap
    return _MIC_CLASS


class MacMicSource(Source):
    """Микрофон по умолчанию на Mac (Системные настройки → Звук → Ввод), через AVFoundation."""

    def __init__(self) -> None:
        super().__init__()
        self.session = None
        self.tap = None

    def start(self) -> None:
        import AVFoundation as AV
        import CoreMedia as CM
        import dispatch

        self.cm = CM
        kind = AV.AVMediaTypeAudio
        status = AV.AVCaptureDevice.authorizationStatusForMediaType_(kind)
        if status == 0:  # про разрешение ещё не спрашивали: macOS покажет окно, ждём ответ
            answered = threading.Event()
            reply: dict = {}

            def on_answer(granted) -> None:
                reply["granted"] = granted
                answered.set()

            AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(kind, on_answer)
            answered.wait(120)
            status = 3 if reply.get("granted") else 2
        if status != 3:
            raise RecorderError(
                "Нет доступа к микрофону. Включите Transcribe в Системные настройки → Конфиденциальность и "
                "безопасность → Микрофон и откройте приложение заново."
            )
        device = AV.AVCaptureDevice.defaultDeviceWithMediaType_(kind)
        if device is None:
            raise RecorderError("Микрофон не найден. Подключите его или выберите в Системные настройки → Звук → Ввод.")
        device_input, error = AV.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if device_input is None:
            raise RecorderError(f"Не удалось включить микрофон. Подробности: {str(error)[:100]}")
        self.session = AV.AVCaptureSession.alloc().init()
        output = AV.AVCaptureAudioDataOutput.alloc().init()
        output.setAudioSettings_(  # просим сразу моно 16 кГц, дробные числа: macOS переводит звук сам и качественно
            {
                "AVFormatIDKey": 1819304813,  # linear PCM
                "AVSampleRateKey": float(RATE),
                "AVNumberOfChannelsKey": 1,
                "AVLinearPCMBitDepthKey": 32,
                "AVLinearPCMIsFloatKey": True,
                "AVLinearPCMIsNonInterleaved": False,
                "AVLinearPCMIsBigEndianKey": False,
            }
        )
        self.tap = _mic_class().alloc().initWithOwner_(self)
        self.queue = dispatch.dispatch_queue_create(b"transcribe.microphone", None)
        output.setSampleBufferDelegate_queue_(self.tap, self.queue)
        if not (self.session.canAddInput_(device_input) and self.session.canAddOutput_(output)):
            raise RecorderError("Не удалось подключить микрофон к записи.")
        self.session.addInput_(device_input)
        self.session.addOutput_(output)
        self.session.startRunning()

    def on_buffer(self, buffer) -> None:
        try:
            self.push(buffer_to_mono(self.cm, buffer))
        except Exception as error:
            self.error = f"Не удалось разобрать звук микрофона: {str(error)[:100]}"

    def stop(self) -> None:
        if self.session is not None:
            self.session.stopRunning()


class PortAudioMicSource(Source):
    """Микрофон по умолчанию на Windows и Linux, через sounddevice. На Mac не используется: там PortAudio зависает."""

    def __init__(self) -> None:
        super().__init__()
        self.stream = None

    def start(self) -> None:
        import sounddevice as sd  # noqa: PLC0415

        def callback(indata, frames, when, status) -> None:
            self.push(to_rate(indata[:, 0].copy(), self.rate))

        self.rate = RATE
        try:
            try:
                self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=callback)
            except sd.PortAudioError:
                # Не каждое устройство умеет 16 кГц сразу (так бывает на Windows): берём его частоту и переводим сами.
                self.rate = int(sd.query_devices(kind="input")["default_samplerate"])
                self.stream = sd.InputStream(samplerate=self.rate, channels=1, dtype="float32", callback=callback)
            self.stream.start()
        except Exception as error:
            raise RecorderError(
                "Не удалось включить микрофон. Проверьте, что он подключён, и что программе разрешён доступ: "
                "Системные настройки → Конфиденциальность и безопасность → Микрофон. "
                f"Подробности: {str(error)[:100]}"
            ) from error

    def stop(self) -> None:
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass


_TAP_CLASS = None


def _tap_class():
    """Класс-посредник между macOS и Python. Создаётся один раз: второй раз тот же класс объявить нельзя."""
    global _TAP_CLASS
    if _TAP_CLASS is None:
        import objc
        from Foundation import NSObject

        class AudioTap(NSObject):
            def initWithOwner_(self, owner):
                self = objc.super(AudioTap, self).init()
                if self is not None:
                    self.owner = owner
                return self

            def stream_didOutputSampleBuffer_ofType_(self, stream, buffer, kind):
                self.owner.on_buffer(buffer, kind)

            def stream_didStopWithError_(self, stream, error):
                self.owner.on_stopped(error)

        _TAP_CLASS = AudioTap
    return _TAP_CLASS


class SystemSource(Source):
    """Звук, который компьютер воспроизводит: речь собеседника в звонке, видео, любой другой звук. Только Mac."""

    def __init__(self) -> None:
        super().__init__()
        self.stream = None
        self.tap = None
        self.stopping = False

    def start(self) -> None:
        if sys.platform != "darwin":
            raise RecorderError("Запись звука из компьютера сейчас работает только на Mac.")
        import CoreMedia as CM
        import ScreenCaptureKit as SCK

        self.sck, self.cm = SCK, CM
        got = threading.Event()
        box: dict = {}

        def on_content(content, error) -> None:
            box["content"], box["error"] = content, error
            got.set()

        SCK.SCShareableContent.getShareableContentWithCompletionHandler_(on_content)
        got.wait(30)
        content = box.get("content")
        if content is None or not content.displays():
            raise RecorderError(
                "Нет доступа к звуку компьютера. Включите Transcribe в Системные настройки → Конфиденциальность и "
                "безопасность → Запись экрана и системного звука и откройте приложение заново. "
                "Экран не записывается, берётся только звук."
            )
        config = SCK.SCStreamConfiguration.alloc().init()
        config.setCapturesAudio_(True)
        config.setSampleRate_(48000)
        config.setChannelCount_(2)
        config.setExcludesCurrentProcessAudio_(True)  # звук самой программы не пишем
        config.setWidth_(2)  # картинка не нужна: делаем её крошечной и не подключаем получателя
        config.setHeight_(2)
        config.setMinimumFrameInterval_(CM.CMTimeMake(1, 1))
        screen_filter = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(content.displays()[0], [])
        self.tap = _tap_class().alloc().initWithOwner_(self)
        self.stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(screen_filter, config, self.tap)
        ok, error = self.stream.addStreamOutput_type_sampleHandlerQueue_error_(self.tap, SCK.SCStreamOutputTypeAudio, None, None)
        if not ok:
            raise RecorderError(f"Не удалось подключиться к звуку компьютера. Подробности: {str(error)[:100]}")
        started = threading.Event()
        result: dict = {}

        def on_start(error) -> None:
            result["error"] = error
            started.set()

        self.stream.startCaptureWithCompletionHandler_(on_start)
        started.wait(15)
        if result.get("error") is not None or not started.is_set():
            raise RecorderError(f"Не удалось начать запись звука компьютера. Подробности: {str(result.get('error'))[:100]}")

    def on_buffer(self, buffer, kind) -> None:
        if kind != self.sck.SCStreamOutputTypeAudio:
            return
        try:
            self.push(buffer_to_mono(self.cm, buffer))
        except Exception as error:
            self.error = f"Не удалось разобрать звук компьютера: {str(error)[:100]}"

    def on_stopped(self, error) -> None:
        if not self.stopping:
            self.error = "Запись звука компьютера прервалась: macOS остановила её. Начните запись заново."

    def stop(self) -> None:
        self.stopping = True
        if self.stream is not None:
            done = threading.Event()
            self.stream.stopCaptureWithCompletionHandler_(lambda error: done.set())
            done.wait(5)


class Recorder:
    """Запись одного или двух источников в один файл. Время в файле идёт по часам компьютера."""

    def __init__(self, mode: str, folder: Path | None = None) -> None:
        if mode not in ("mic", "system", "both"):
            raise ValueError(mode)
        self.mode = mode
        self.folder = folder or DEFAULT_FOLDER
        self.sources: dict[str, Source] = {}
        self.path: Path | None = None
        self.started = 0.0
        self.thread: threading.Thread | None = None
        self.finish = threading.Event()
        self.stopped_at = 0.0  # когда попросили закончить: файл должен быть ровно такой длины, сколько шла запись
        self.writer: WavWriter | None = None

    def start(self) -> Path:
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise RecorderError(f"Не удалось создать папку для записей {self.folder}. Подробности: {str(error)[:100]}") from error
        name = f"Запись {datetime.now():%Y-%m-%d %H-%M-%S}"
        self.path = self.folder / f"{name}.wav"
        n = 2
        while self.path.exists():
            self.path = self.folder / f"{name}-{n}.wav"
            n += 1
        wanted = {"mic": ["mic"], "system": ["system"], "both": ["system", "mic"]}[self.mode]
        try:
            for kind in wanted:
                source = SystemSource() if kind == "system" else (MacMicSource() if sys.platform == "darwin" else PortAudioMicSource())
                source.start()
                self.sources[kind] = source
        except RecorderError:
            self._stop_sources()
            raise
        self.writer = WavWriter(self.path)
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self.path

    def _run(self) -> None:
        written = 0
        while True:
            last = self.finish.wait(0.2)
            if last:
                chunks = [source.take(None) for source in self.sources.values()]
                due = int((self.stopped_at - self.started) * RATE) - written  # файл должен дотянуться до момента остановки
                length = max([due] + [len(c) for c in chunks])
            else:
                length = int((time.monotonic() - self.started - LAG) * RATE) - written
                if length <= 0:
                    continue
                chunks = [source.take(length) for source in self.sources.values()]
            mix = np.zeros(length, np.float32)
            for chunk in chunks:
                mix[: len(chunk)] += chunk[:length]
            self.writer.write(mix)
            written += length
            if last:
                return

    def _stop_sources(self) -> None:
        """Останавливает источники, каждый в своём потоке и не дольше 4 секунд: если системный звук или микрофон
        «зависнет» на остановке, файл всё равно будет дописан и закрыт."""
        threads = [threading.Thread(target=source.stop, daemon=True) for source in self.sources.values()]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=4)

    def stop(self) -> Path | None:
        """Заканчивает запись, дописывает хвост и закрывает файл."""
        self.stopped_at = time.monotonic()
        self._stop_sources()
        self.finish.set()
        if self.thread:
            self.thread.join(timeout=10)
        if self.writer:
            self.writer.close()
            self.writer = None
        return self.path

    def elapsed(self) -> float:
        return time.monotonic() - self.started if self.started else 0.0

    def level(self) -> float:
        """Громкость от 0 до 1 за время с прошлого опроса (по шкале децибел: тихая речь заметна)."""
        peak = max((source.level() for source in self.sources.values()), default=0.0)
        if peak <= 0:
            return 0.0
        return float(min(1.0, max(0.0, (20 * np.log10(peak) + 60) / 60)))

    def problem(self) -> str | None:
        """Сообщение, если что-то пошло не так прямо во время записи."""
        for source in self.sources.values():
            if source.error:
                return source.error
        mic = self.sources.get("mic")
        if mic and mic.silent_since and time.monotonic() - mic.silent_since > 3 and self.elapsed() > 3:
            return (
                "Микрофон не передаёт звук. Проверьте, что он не выключен и что приложению разрешён доступ: "
                "Системные настройки → Конфиденциальность и безопасность → Микрофон."
            )
        system = self.sources.get("system")
        if system and system.silent_since and time.monotonic() - system.silent_since > 10 and self.elapsed() > 10:
            # Без разрешения macOS не сообщает об ошибке, а присылает тишину: от настоящей паузы в разговоре это не отличить.
            return (
                "Звук из компьютера пока не слышен. Если звонок идёт, проверьте, что Transcribe включён в "
                "Системные настройки → Конфиденциальность и безопасность → Запись экрана и системного звука."
            )
        return None


def analyze(path: Path) -> dict:
    """Читает готовый WAV и возвращает длительность, громкость и главную частоту: для проверки."""
    import wave

    with wave.open(str(path), "rb") as w:
        rate, frames = w.getframerate(), w.getnframes()
        data = np.frombuffer(w.readframes(frames), "<i2").astype(np.float32) / 32768
    result = {"seconds": round(frames / rate, 2), "rate": rate, "peak": round(float(np.abs(data).max()), 4) if frames else 0.0}
    result["rms"] = round(float(np.sqrt((data**2).mean())), 5) if frames else 0.0
    if frames > rate:
        spectrum = np.abs(np.fft.rfft(data[: rate * (frames // rate)]))
        result["main_hz"] = int(np.argmax(spectrum) * rate / (rate * (frames // rate)))
    return result


def selftest(mode: str, seconds: float, out: Path | None) -> dict:
    """Записывает несколько секунд и сообщает, что получилось. Без окна: для проверки программы и собранного приложения."""
    recorder = Recorder(mode, out.parent if out else None)
    try:
        path = recorder.start()
        time.sleep(seconds)
        problem = recorder.problem()
        recorder.stop()
    except RecorderError as error:
        return {"ok": False, "error": str(error)}
    if out:
        path.replace(out)
        path = out
    # received — сколько секунд звука реально пришло от каждого источника: ноль значит, что звук не поступал вовсе
    received = {name: round(source.received / RATE, 1) for name, source in recorder.sources.items()}
    return {"ok": True, "file": str(path), "problem": problem, "received_seconds": received, **analyze(path)}


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2 or sys.argv[1] not in ("mic", "system", "both"):
        print("Использование: uv run recorder.py mic|system|both [секунды] [файл.wav]")
        sys.exit(1)
    target = Path(sys.argv[3]).expanduser() if len(sys.argv) > 3 else None
    print(json.dumps(selftest(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 4, target), ensure_ascii=False, indent=2))
