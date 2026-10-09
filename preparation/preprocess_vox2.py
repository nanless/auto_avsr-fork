import argparse
import json
import math
import os
import pickle
import shutil
import sys
import warnings

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from utils import save_vid_aud


def _clip_is_complete(vid_filename, dst_vid_dir):
    """True only when EVERY segment index of a clip is a complete video+audio PAIR.

    The probe used to check just `<clip>_00.mp4`.  Two gaps followed from that:

      * save_vid_aud writes the video and then the audio straight to their final
        names (no .part + rename), so a worker killed between the two writes
        leaves a decodable `_00.mp4` with no wav.  Every later resume counted the
        whole clip as done and the incomplete pair could never be repaired.
      * any index above 0 was never checked at all, so a torn `_01.mp4` was
        invisible too.

    Scan the mirror directory for every `<clip>_NN.mp4`, require the matching
    `.wav`, and validate each with _segment_is_complete.  A clip with no segments
    at all is not complete (it still needs work).
    """
    import glob as _glob
    base = vid_filename.replace(args.vid_dir, dst_vid_dir)[:-4]
    vids = sorted(_glob.glob(_glob.escape(base) + "_[0-9][0-9].mp4"))
    if not vids:
        return False
    for v in vids:
        w = v[:-4] + ".wav"
        if not os.path.exists(w) or os.path.getsize(w) < 1024:
            return False
        if not _segment_is_complete(v):
            return False
        if not _segment_is_complete_audio(w):
            return False
    return True


def _segment_is_complete_audio(path):
    """Same idea as _segment_is_complete, for the wav half of a segment pair."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 1024:
        return False
    try:
        import subprocess
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=duration", "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return bool(out) and float(out.splitlines()[0]) > 0
    except Exception:
        return False


def _segment_is_complete(path):
    """True only when a previously written segment is actually usable.

    The resume probe used to be `os.path.exists`, but the segment writer
    (torchvision.io.write_video) opens the FINAL path directly, so a worker killed
    mid-encode leaves a truncated mp4 behind.  Existence then reads as success and
    the clip is skipped forever, silently shrinking the corpus with no way to
    repair it short of deleting the file by hand.  Require a decodable video
    stream of non-zero duration instead.
    """
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 1024:
        return False
    try:
        import subprocess
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration", "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return bool(out) and float(out.splitlines()[0]) > 0
    except Exception:
        return False



warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser(description="VoxCeleb2 Preprocessing")
parser.add_argument(
    "--vid-dir",
    type=str,
    required=True,
    help="Directory where the video sequence is stored",
)
parser.add_argument(
    "--aud-dir",
    type=str,
    required=True,
    help="Directory where the audio sequence is stored",
)
parser.add_argument(
    "--label-dir",
    type=str,
    default="",
    help="Directory where lid.csv is saved",
)
parser.add_argument(
    "--landmarks-dir",
    type=str,
    default=None,
    help="Directory of landmarks",
)
parser.add_argument(
    "--detector",
    type=str,
    help="Type of face detector",
)
parser.add_argument(
    "--root-dir",
    type=str,
    required=True,
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="vox2",
    help="Name of dataset",
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=16,
    help="Max duration (second) for each segment, (Default: 16)",
)
parser.add_argument(
    "--combine-av",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Merges the audio and video components to a media file",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads to be used in parallel",
)
parser.add_argument(
    "--job-index",
    type=int,
    default=0,
    help="Index to identify separate jobs (useful for parallel processing)",
)
args = parser.parse_args()


# Constants
seg_vid_len = args.seg_duration * 25
seg_aud_len = args.seg_duration * 16000
dst_vid_dir = os.path.join(
    args.root_dir, args.dataset, f"{args.dataset}_video_seg{args.seg_duration}s"
)

# Load data
vid_dataloader = AVSRDataLoader(
    modality="video", detector=args.detector, convert_gray=False
)
aud_dataloader = AVSRDataLoader(modality="audio")

# Load video and audio files
filenames = [
    os.path.join(args.vid_dir, _ + ".mp4")
    for _ in open(os.path.join(args.label_dir, "vox-en.id")).read().splitlines()
]

unit = math.ceil(len(filenames) / args.groups)
files_to_process = filenames[args.job_index * unit : (args.job_index + 1) * unit]

# The widened except below is deliberate -- AVSpeech sources are YouTube scrapes
# and some really are unreadable -- but a bare `continue` made a SYSTEMATIC
# failure indistinguishable from a clean run: every clip skipped, no output, and
# the process still exited 0.  Record every skip with its cause and report
# counts at the end so a shrinking corpus is visible.
_skip_log = open(os.path.join(args.root_dir, f"skipped_job{args.job_index}.jsonl"), "a", encoding="utf-8")
_n_done = _n_skipped = _n_segments = _n_attempted = 0
_skip_reasons = {}

def _record_skip(clip, reason, detail=""):
    global _n_skipped
    _n_skipped += 1
    _skip_reasons[reason] = _skip_reasons.get(reason, 0) + 1
    _skip_log.write(json.dumps({
        "clip": os.path.relpath(clip, args.vid_dir) if clip.startswith(args.vid_dir) else clip,
        "reason": reason, "detail": detail[:300],
    }, ensure_ascii=False) + "\n")
    _skip_log.flush()

for vid_filename in tqdm(files_to_process):
    # Resume support: a run that was interrupted (or a worker restarted after a
    # crash) must not redo clips whose first segment already exists.  Only the
    # bookkeeping changes -- the per-frame RetinaFace/FAN path is untouched.
    if _clip_is_complete(vid_filename, dst_vid_dir):
        _n_done += 1
        continue
    if args.landmarks_dir:
        landmarks_filename = (
            vid_filename.replace(args.vid_dir, args.landmarks_dir)[:-4] + ".pkl"
        )
        landmarks = pickle.load(open(landmarks_filename, "rb"))
    else:
        landmarks = None
    try:
        video_data = vid_dataloader.load_data(vid_filename, landmarks)
        aud_filename = vid_filename.replace(args.vid_dir, args.aud_dir)[:-4] + ".wav"
        audio_data = aud_dataloader.load_data(aud_filename)
    except (
        UnboundLocalError,
        TypeError,
        OverflowError,
        AssertionError,
        RuntimeError,      # torchaudio/ffmpeg raise this for unreadable or missing media
        OSError,           # missing file on the filesystem
        ValueError,
    ) as _exc:
        _record_skip(vid_filename, type(_exc).__name__, repr(_exc))
        continue
    if video_data is None:
        _record_skip(vid_filename, "video_data_none")
        continue

    # Process segments
    for i, start_idx in enumerate(range(0, len(video_data), seg_vid_len)):
        dst_vid_filename = (
            f"{vid_filename.replace(args.vid_dir, dst_vid_dir)[:-4]}_{i:02d}.mp4"
        )
        dst_aud_filename = (
            f"{aud_filename.replace(args.aud_dir, dst_vid_dir)[:-4]}_{i:02d}.wav"
        )
        # NOTE: do NOT count here.  This loop iterates over candidate 16 s windows,
        # not over segments actually written, so incrementing before the gates made
        # `segments_written` overstate the output -- a clip that loaded but wrote
        # nothing still reported segments_written > 0, recorded no skip, and passed
        # the `_n_segments == 0 and _n_done == 0` guard below.  A multi-agent audit
        # found 11 already-processed clips lost that way with no trace anywhere.
        _n_attempted += 1
        trim_video_data = video_data[start_idx : start_idx + seg_vid_len]
        trim_audio_data = audio_data[
            :, start_idx * 640 : (start_idx + seg_vid_len) * 640
        ]
        if trim_video_data is None or trim_audio_data is None:
            _record_skip(vid_filename, "segment_slice_none", f"index={i}")
            continue
        video_length = len(trim_video_data)
        audio_length = trim_audio_data.size(1)
        if (
            audio_length / video_length < 560.0
            or audio_length / video_length > 720.0
            or video_length < 12
        ):
            # Record it: without this the clip vanished with no skip record and no
            # segment, so a systematic short-audio regression would shrink the
            # corpus while every worker exited 0 and every gate reported success.
            _record_skip(
                vid_filename,
                "ratio_gate",
                f"index={i} audio={audio_length} video={video_length} "
                f"ratio={audio_length / max(video_length, 1):.1f}",
            )
            continue

        # Save video and audio
        save_vid_aud(
            dst_vid_filename,
            dst_aud_filename,
            trim_video_data,
            trim_audio_data,
            video_fps=25,
            audio_sample_rate=16000,
        )
        # Count only segments actually written, so the summary and the
        # `_n_segments == 0` guard describe real output.
        _n_segments += 1

        # Merge video and audio
        if args.combine_av:
            in1 = ffmpeg.input(dst_vid_filename)
            in2 = ffmpeg.input(dst_aud_filename)
            out = ffmpeg.output(
                in1["v"],
                in2["a"],
                dst_vid_filename[:-4] + ".m.mp4",
                vcodec="copy",
                acodec="aac",
                strict="experimental",
                loglevel="panic",
            )
            out.run()
            os.remove(dst_aud_filename)
            os.remove(dst_vid_filename)
            shutil.move(dst_vid_filename[:-4] + ".m.mp4", dst_vid_filename)

_skip_log.close()
_summary = {
    "job_index": args.job_index,
    "input_clips": len(files_to_process),
    "already_complete": _n_done,
    "skipped": _n_skipped,
    "segments_written": _n_segments,
    "segments_attempted": _n_attempted,
    "skip_reasons": dict(sorted(_skip_reasons.items(), key=lambda kv: -kv[1])),
}
print("PREPROCESS_SUMMARY " + json.dumps(_summary, ensure_ascii=False), flush=True)

# A run that skipped everything is a failure, not a success.  Exit non-zero so the
# pipeline's wait_pids sees it.
if _n_segments == 0 and _n_done == 0:
    sys.exit(3)
