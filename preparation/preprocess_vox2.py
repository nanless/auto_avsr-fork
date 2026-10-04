import argparse
import math
import os
import pickle
import shutil
import warnings

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from utils import save_vid_aud


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

for vid_filename in tqdm(files_to_process):
    # Resume support: a run that was interrupted (or a worker restarted after a
    # crash) must not redo clips whose first segment already exists.  Only the
    # bookkeeping changes -- the per-frame RetinaFace/FAN path is untouched.
    _probe = f"{vid_filename.replace(args.vid_dir, dst_vid_dir)[:-4]}_00.mp4"
    if _segment_is_complete(_probe):
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
    ):
        continue
    if video_data is None:
        continue

    # Process segments
    for i, start_idx in enumerate(range(0, len(video_data), seg_vid_len)):
        dst_vid_filename = (
            f"{vid_filename.replace(args.vid_dir, dst_vid_dir)[:-4]}_{i:02d}.mp4"
        )
        dst_aud_filename = (
            f"{aud_filename.replace(args.aud_dir, dst_vid_dir)[:-4]}_{i:02d}.wav"
        )
        trim_video_data = video_data[start_idx : start_idx + seg_vid_len]
        trim_audio_data = audio_data[
            :, start_idx * 640 : (start_idx + seg_vid_len) * 640
        ]
        if trim_video_data is None or trim_audio_data is None:
            continue
        video_length = len(trim_video_data)
        audio_length = trim_audio_data.size(1)
        if (
            audio_length / video_length < 560.0
            or audio_length / video_length > 720.0
            or video_length < 12
        ):
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
