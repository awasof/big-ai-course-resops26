#!/bin/bash
# Собирает приложение для Mac: dist/Transcribe.app (двойной щелчок, окно, без терминала).
# Нужен только uv. Запуск из этой папки: ./build-mac.sh
# Приложение собирается под этот Mac (Apple Silicon или Intel — какой у вас). Для Windows нужна сборка на Windows.
set -e
cd "$(dirname "$0")"
WORK="$(mktemp -d)"

# Иконка: icon.icns лежит рядом (рисует её make_icon.py). Без неё приложение соберётся со стандартной иконкой.
ICON=()
if [ -f icon.icns ]; then
  ICON=(--icon "$PWD/icon.icns")
fi

# recorder.py лежит рядом с transcribe_app.py, и PyInstaller подхватывает его сам по команде import.
# Микрофон и звук компьютера берутся средствами macOS через pyobjc: ScreenCaptureKit, AVFoundation, CoreMedia, libdispatch.
uv run --no-project --with pyinstaller --with "faster-whisper>=1.1" --with "av<19" --with numpy \
  --with pyobjc-framework-ScreenCaptureKit --with pyobjc-framework-CoreMedia \
  --with pyobjc-framework-AVFoundation --with pyobjc-framework-libdispatch \
  pyinstaller --noconfirm --windowed --name Transcribe --osx-bundle-identifier local.practice.transcribe "${ICON[@]}" \
  --distpath dist --workpath "$WORK/build" --specpath "$WORK" \
  --collect-all faster_whisper --collect-all ctranslate2 --collect-all onnxruntime --collect-all av \
  --hidden-import ScreenCaptureKit --hidden-import CoreMedia --hidden-import AVFoundation --hidden-import dispatch \
  --hidden-import objc --hidden-import Foundation \
  "$PWD/transcribe_app.py"

rm -rf "$WORK" dist/Transcribe  # папка dist/Transcribe — промежуточный дубль приложения, нужен только .app

# Строка NSMicrophoneUsageDescription — то, что macOS показывает в окне с вопросом про микрофон; по документации Apple
# приложению без неё доступ к микрофону не выдаётся. Без этой строки сборку не пробовал.
APP="$PWD/dist/Transcribe.app"
PLIST="$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Delete :NSMicrophoneUsageDescription" "$PLIST" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :NSMicrophoneUsageDescription string Transcribe записывает звук с микрофона, чтобы расшифровать его в текст. Запись остаётся на этом компьютере." "$PLIST"
# Правка Info.plist ломает подпись пакета: подписываем заново (подпись «на месте», не сертификатом Apple).
codesign --force --deep --sign - "$APP"

echo
echo "Готово: $APP"
