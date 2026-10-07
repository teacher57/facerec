#!/usr/bin/env bash
# Sets everything up on a Mac, Linux PC, or Raspberry Pi:
# Python environment, packages, the two pretrained models, and the `facerec` command.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_BASE="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
MODELS=(
  "face_detection_yunet/face_detection_yunet_2023mar.onnx"
  "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

is_raspberry_pi() {
  [ -f /proc/device-tree/model ] && grep -qi "raspberry pi" /proc/device-tree/model
}

echo "==> Python environment"
if is_raspberry_pi; then
  echo "Raspberry Pi detected: installing picamera2 from apt (it can't be installed with pip)."
  sudo apt-get update
  sudo apt-get install -y python3-picamera2 python3-venv
  # --system-site-packages lets the venv see the apt-installed picamera2.
  python3 -m venv --system-site-packages .venv
  # OpenCV 4.8-4.9 supports these models and works with the system NumPy 1.x that
  # picamera2 is built against (newer OpenCV pulls in NumPy 2).
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install "opencv-python>=4.8,<4.10"
else
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install opencv-python numpy
fi

echo "==> Models"
mkdir -p models
for model in "${MODELS[@]}"; do
  file="models/$(basename "$model")"
  if [ -f "$file" ]; then
    echo "already have $file"
    continue
  fi
  echo "downloading $file"
  curl -fL --progress-bar -o "$file" "$MODEL_BASE/$model"
  # A tiny file means we got a Git LFS pointer instead of the real model.
  if [ "$(wc -c < "$file")" -lt 100000 ]; then
    echo "error: $file is too small to be the real model" >&2
    rm -f "$file"
    exit 1
  fi
done

echo "==> facerec command (for all users, in /usr/local/bin; asks for your password)"
rm -f face  # old launcher from earlier versions
REPO="$(pwd)"
LAUNCHER="$(mktemp)"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
# Installed by $(printf '%q' "$REPO/setup.sh"); re-run it if the project folder moves.
exec $(printf '%q' "$REPO/.venv/bin/python") $(printf '%q' "$REPO/face.py") "\$@"
EOF
if sudo mkdir -p /usr/local/bin && sudo install -m 755 "$LAUNCHER" /usr/local/bin/facerec; then
  rm -f "$LAUNCHER"
  echo "installed /usr/local/bin/facerec"
else
  rm -f "$LAUNCHER"
  echo "Could not install /usr/local/bin/facerec (sudo failed). Run ./setup.sh again to retry." >&2
  exit 1
fi

echo
echo "Done. Try:"
echo "  facerec help"
echo "  facerec enroll_face --from-camera --name YourName"
echo "  facerec test -v -j"
