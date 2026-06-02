#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${PYTHON:-python3}
VENV="$ROOT/.venv"
EXTRAS=${GAZE_EXTRAS:-parquet}

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [--extras none|parquet|dev] [--venv PATH]

Options:
  --extras none      Install only the core CLI. Tables fall back to JSONL.
  --extras parquet   Install runtime Parquet dependencies. Default.
  --extras dev       Install Parquet dependencies plus test tooling.
  --venv PATH        Create/use a virtualenv at PATH. Default: .venv

Environment:
  PYTHON             Python executable to use. Default: python3
  GAZE_EXTRAS        Default extras group if --extras is omitted.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --extras)
      EXTRAS=$2
      shift 2
      ;;
    --venv)
      VENV=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$EXTRAS" in
  none|parquet|dev) ;;
  *)
    echo "--extras must be one of: none, parquet, dev" >&2
    exit 2
    ;;
esac

echo "Using repository: $ROOT"
echo "Checking Python..."
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required")
print(sys.version.split()[0])
PY

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "warning: ffmpeg not found; install ffmpeg for real video resampling/transcoding" >&2
fi

if [ ! -d "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
fi

if [ "$EXTRAS" = "none" ]; then
  "$VENV/bin/python" -m pip install --no-build-isolation -e "$ROOT"
else
  "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
  "$VENV/bin/python" -m pip install -e "$ROOT[$EXTRAS]"
fi
"$VENV/bin/gaze" doctor

echo "Setup complete. Activate with: . $VENV/bin/activate"
