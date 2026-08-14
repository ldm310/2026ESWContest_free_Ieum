#!/usr/bin/env bash

# Jetson에서 실시간 DOA·MVDR 처리와 자막 UI를 한 번에 실행한다.
set -Eeuo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BEM_TABLE="${BEM_TABLE:-$PROJECT_DIR/bem_table_reduced.h5}"
AUDIO_DEVICE="${AUDIO_DEVICE:-ReSpeaker}"
CAMERA_MODE="${CAMERA_MODE:-auto}"
UVC_DEVICE="${UVC_DEVICE:-0}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-8765}"
NO_BROWSER="${NO_BROWSER:-0}"
DOA_PID=""
UI_ARGS=(--host "$UI_HOST" --port "$UI_PORT")

if [[ "$NO_BROWSER" == "1" ]]; then
  UI_ARGS+=(--no-browser)
fi

cleanup() {
  if [[ -n "$DOA_PID" ]] && kill -0 "$DOA_PID" 2>/dev/null; then
    kill -TERM "$DOA_PID" 2>/dev/null || true
    wait "$DOA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$PROJECT_DIR"

if [[ "${1:-}" == "--demo" ]]; then
  exec "$PYTHON_BIN" stage6_caption_ui.py --demo "${UI_ARGS[@]}"
fi

if [[ ! -f "$BEM_TABLE" ]]; then
  echo "[실행 오류] BEM 테이블을 찾을 수 없습니다: $BEM_TABLE" >&2
  exit 1
fi

"$PYTHON_BIN" realtime_doa.py \
  --bem-table "$BEM_TABLE" \
  --device "$AUDIO_DEVICE" &
DOA_PID=$!

"$PYTHON_BIN" stage6_caption_ui.py \
  "${UI_ARGS[@]}" \
  --camera "$CAMERA_MODE" \
  --uvc-device "$UVC_DEVICE"
