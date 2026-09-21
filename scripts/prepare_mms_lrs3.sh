#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/code/github_repos/auto_avsr}"
RAW_ROOT="${RAW_ROOT:-/root/group-shared/voiceprint/data/multimodal/multimodal/audio-visual/preprocessed_data/lrs3}"
LANDMARKS_ROOT="${LANDMARKS_ROOT:-/root/group-shared/voiceprint/data/multimodal/multimodal/audio-visual/derived/avsr_reproduction/auto_avsr_landmarks/LRS3_landmarks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/group-shared/voiceprint/data/multimodal/multimodal/audio-visual/derived/avsr_reproduction/mms_lrs3_24s}"
WORKERS="${WORKERS:-24}"
ENV_PYTHON="${ENV_PYTHON:-/root/miniforge3/envs/auto-avsr-prep/bin/python}"

if ! [[ "${WORKERS}" =~ ^[0-9]+$ ]] || (( WORKERS < 1 || WORKERS > 32 )); then
  echo "WORKERS must be an integer in [1, 32]" >&2
  exit 2
fi

raw_real="$(realpath "${RAW_ROOT}")"
mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/locks"
output_real="$(realpath "${OUTPUT_ROOT}")"
case "${output_real}/" in
  "${raw_real}/"*) echo "Refusing to write under the raw LRS3 root" >&2; exit 2 ;;
esac

for split in pretrain trainval test; do
  test -d "${RAW_ROOT}/${split}"
  test -d "${LANDMARKS_ROOT}/${split}"
done

landmark_count="$(find "${LANDMARKS_ROOT}" -type f -name '*.pkl' | wc -l)"
if (( landmark_count < 151000 )); then
  echo "Landmark archive looks incomplete: ${landmark_count} .pkl files" >&2
  exit 3
fi

exec 9>"${OUTPUT_ROOT}/locks/prepare.lock"
if ! flock -n 9; then
  echo "Another MMS LRS3 preparation process is already running" >&2
  exit 4
fi

export PYTHONNOUSERSITE=1
cd "${REPO_ROOT}/preparation"

run_subset() {
  local subset="$1"
  local -a pids=()
  local job
  for ((job=0; job<WORKERS; job++)); do
    "${ENV_PYTHON}" preprocess_lrs2lrs3.py \
      --data-dir "${RAW_ROOT}" \
      --landmarks-dir "${LANDMARKS_ROOT}" \
      --root-dir "${OUTPUT_ROOT}" \
      --subset "${subset}" \
      --dataset lrs3 \
      --gpu_type cuda \
      --seg-duration 24 \
      --combine-av false \
      --groups "${WORKERS}" \
      --job-index "${job}" \
      >"${OUTPUT_ROOT}/logs/${subset}.${WORKERS}.${job}.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "At least one ${subset} shard failed; inspect ${OUTPUT_ROOT}/logs" >&2
    return 1
  fi
}

run_subset train
run_subset test

echo "MMS LRS3 24-second preprocessing completed: ${OUTPUT_ROOT}"
