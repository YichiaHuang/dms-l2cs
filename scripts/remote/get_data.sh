#!/bin/bash
# Fetch MPIIFaceGaze and convert it to the GazeHub layout that train_mpii.py reads.
#   bash scripts/remote/get_data.sh                  # -> ~/datasets/raw/MPIIFaceGaze (zip + extracted), ~/datasets/MPIIFaceGaze (processed)
#   DATA_ROOT=/work/$USER/datasets bash scripts/remote/get_data.sh
# If the processed dataset was copied from another machine (scp/rsync of ~/datasets/MPIIFaceGaze), this script
# just verifies it and exits. Expected result: Label/p00.label .. p14.label, 37,667 face images in Image/pXX/face/.
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets}"
RAW="$DATA_ROOT/raw/MPIIFaceGaze"
OUT="$DATA_ROOT/MPIIFaceGaze"
URL="https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip"   # ~900 MB, MPI-INF; licence: research only

verify() {
  local n_label n_face
  n_label=$(ls "$OUT/Label" 2>/dev/null | wc -l)
  n_face=$(find "$OUT/Image" -path '*/face/*.jpg' 2>/dev/null | wc -l)
  echo "labels: $n_label (expect 15), face images: $n_face (expect 37667)"
  [[ "$n_label" -eq 15 && "$n_face" -ge 37000 ]]
}

if [[ -d "$OUT/Label" ]] && verify; then echo "processed dataset already present at $OUT"; exit 0; fi

mkdir -p "$RAW"
if [[ ! -f "$RAW/MPIIFaceGaze.zip" ]]; then
  echo "== downloading $URL"
  curl -L --retry 5 -C - -o "$RAW/MPIIFaceGaze.zip.part" "$URL" && mv "$RAW/MPIIFaceGaze.zip.part" "$RAW/MPIIFaceGaze.zip"
fi
if [[ ! -d "$RAW/MPIIFaceGaze/p00" ]]; then
  echo "== extracting"
  python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$RAW/MPIIFaceGaze.zip" "$RAW"
fi
ls "$RAW/MPIIFaceGaze" | head -3
echo "== preprocessing (GazeHub normalisation, parallel over persons; ~10-20 min)"
python -I "$ROOT/scripts/prep_mpiifacegaze.py" --raw "$RAW/MPIIFaceGaze" --out "$OUT"
verify && echo "dataset ready at $OUT"
