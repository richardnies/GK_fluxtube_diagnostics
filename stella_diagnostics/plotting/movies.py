"""Frame-by-frame movie rendering: the mkdir-frames/skip-if-exists/ffmpeg
skeleton duplicated across every movie_*.py example script, extracted into
one reusable pair of functions.

Decoupled from StellaRun (like other plotting/*.py modules) -- render_movie
only knows about frame indices, a caller-supplied per-frame plotting
callback, and files.
"""

import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


class FFmpegNotFoundError(RuntimeError):
    pass


def ffmpeg_frames_to_video(img_dir, frame_glob, video_path, fps=30, extra_args=None):
    """Encode the PNG frames in `img_dir` matching `frame_glob` into a video
    at `video_path` via ffmpeg.

    Uses subprocess.run (not os.system): avoids shell-string interpolation
    of paths, and surfaces ffmpeg's exit code/stderr instead of discarding
    them. Raises FFmpegNotFoundError with a clear message if the ffmpeg
    binary isn't on PATH, instead of silently doing nothing.

    Uses `-pattern_type glob` rather than a fixed numeric %03d pattern, so
    a gap in the frame sequence (e.g. one frame's plotting call raised and
    was skipped) doesn't truncate the video at the first missing frame --
    a real latent bug in the current numeric-pattern scripts.
    """
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError(
            "ffmpeg not found on PATH -- install it to render movies "
            "(frames were still written to " + str(img_dir) + ")"
        )

    img_dir = Path(img_dir)
    video_path = Path(video_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        str(img_dir / frame_glob),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    if extra_args:
        cmd += list(extra_args)
    cmd.append(str(video_path))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit code {result.returncode}) encoding {video_path}:\n{result.stderr}"
        )
    return video_path


def render_movie(
    img_dir,
    frame_indices,
    frame_fn,
    fps=30,
    rerun_all=False,
    frame_name_fmt="fig_t-%03d.png",
    video_name="video.mp4",
    ffmpeg_extra_args=None,
    on_error="continue",
    verbose=True,
):
    """Render one frame per entry of `frame_indices` via `frame_fn`, then
    encode them into a video with ffmpeg.

    img_dir: frame/video output directory, created if missing (replaces
      ``os.system("mkdir -p " + img_dir)``).
    frame_indices: sequence passed one at a time to frame_fn as (i, idx).
    frame_fn(i, idx) -> matplotlib.figure.Figure, or None to skip this
      frame without treating it as an error.
    rerun_all: False (default) skips frames whose PNG already exists,
      replacing the hand-toggled boolean + exists() check duplicated
      across the movie_*.py scripts. True clears img_dir first (matching
      today's ``rm -rf img_dir/*``) so every frame is regenerated.
    on_error: "continue" (log and skip that frame, matching most existing
      scripts' per-frame try/except) or "raise".

    Each frame's figure is closed immediately after saving -- fixes a
    latent memory-growth bug present in every existing movie script, none
    of which call plt.close() inside a loop that can run hundreds of
    iterations.

    Returns the path to the written video.
    """
    img_dir = Path(img_dir)
    if rerun_all and img_dir.exists():
        for f in img_dir.glob("*.png"):
            f.unlink()
    img_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(frame_indices):
        frame_path = img_dir / (frame_name_fmt % i)
        if not rerun_all and frame_path.exists():
            continue

        if verbose:
            print(f"Rendering frame {i + 1}/{len(frame_indices)}...", end="\r")

        try:
            fig = frame_fn(i, idx)
        except Exception as e:
            if on_error == "raise":
                raise
            print(f"\nSkipping frame {i} ({idx}): {e!r}")
            continue

        if fig is None:
            continue

        fig.savefig(frame_path)
        plt.close(fig)

    if verbose:
        print()

    video_path = img_dir / video_name
    return ffmpeg_frames_to_video(img_dir, "*.png", video_path, fps=fps, extra_args=ffmpeg_extra_args)
