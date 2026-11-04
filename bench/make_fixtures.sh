#!/usr/bin/env bash
# Regenerates the synthetic WAV fixtures from manifest.json using espeak-ng + ffmpeg.
# Output: 16 kHz mono 16-bit, 300 ms leading silence. Real recordings are better; see README.
set -euo pipefail
cd "$(dirname "$0")/fixtures"
python3 - <<'PY' | while IFS=$'\t' read -r file text; do
import json
for f in json.load(open("manifest.json"))["fixtures"]:
    print(f"{f['file']}\t{f['text']}")
PY
  espeak-ng -v en-us -s 165 -w "${TMPDIR:-/tmp}/_fixture.wav" "$text"
  ffmpeg -nostdin -loglevel error -y -i "${TMPDIR:-/tmp}/_fixture.wav" -af "adelay=300" -ar 16000 -ac 1 -sample_fmt s16 "$file"
  echo "wrote $file"
done
