#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/code/github_repos/auto_avsr}"
LANDMARK_ROOT="${LANDMARK_ROOT:-/root/group-shared/voiceprint/data/multimodal/multimodal/audio-visual/derived/avsr_reproduction/auto_avsr_landmarks}"
ARCHIVE="${ARCHIVE:-${LANDMARK_ROOT}/LRS3_landmarks.zip}"
LOG_ROOT="${LOG_ROOT:-/root/code/github_repos/avsr_download_logs}"
DOWNLOAD_PID_FILE="${DOWNLOAD_PID_FILE:-${LOG_ROOT}/lrs3-landmarks-gdrive.pid}"
PREP_LOG="${PREP_LOG:-${LOG_ROOT}/mms-lrs3-24s-preparation.log}"
PREP_PID_FILE="${PREP_PID_FILE:-${LOG_ROOT}/mms-lrs3-24s-preparation.pid}"
PREP_DONE="${PREP_DONE:-${LOG_ROOT}/mms-lrs3-24s-preparation.complete}"

if [[ ! -s "${ARCHIVE}" ]]; then
  test -s "${DOWNLOAD_PID_FILE}"
  download_pid="$(cat "${DOWNLOAD_PID_FILE}")"
  while kill -0 "${download_pid}" 2>/dev/null; do
    echo "WAIT LRS3-landmarks pid=${download_pid}"
    sleep 30
  done
fi

test -s "${ARCHIVE}"
"/root/miniforge3/envs/auto-avsr-prep/bin/python" \
"${REPO_ROOT}/scripts/finalize_landmarks.py" \
  --archive "${ARCHIVE}" \
  --output-root "${LANDMARK_ROOT}"
"/root/code/github_repos/avsr_reproduction/record_asset.sh" \
  "${ARCHIVE}" \
  "https://huggingface.co/datasets/faori/lrs3_landmark/resolve/main/LRS3_landmarks.zip" \
  "19478552863" \
  "f1d2e481c17c9c038989358cde6fe52922ffee58e3833927d9a8ea0dbb6e930b"

if [[ -s "${PREP_PID_FILE}" ]] && kill -0 "$(cat "${PREP_PID_FILE}")" 2>/dev/null; then
  echo "MMS LRS3 preparation already running"
  exit 0
fi

nohup bash -c \
  'WORKERS=24 /root/code/github_repos/auto_avsr/scripts/prepare_mms_lrs3.sh && touch /root/code/github_repos/avsr_download_logs/mms-lrs3-24s-preparation.complete' \
  >"${PREP_LOG}" 2>&1 &
echo $! >"${PREP_PID_FILE}"
echo "STARTED MMS LRS3 preparation pid=$!"
