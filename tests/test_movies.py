"""Data-free tests for stella_diagnostics.plotting.movies: frame-directory
creation, skip-if-exists/rerun_all behavior, error handling, figure cleanup,
and the ffmpeg subprocess call (mocked -- no real ffmpeg dependency)."""
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from stella_diagnostics.plotting.movies import (
    FFmpegNotFoundError,
    ffmpeg_frames_to_video,
    render_movie,
)


def _fake_ffmpeg_run(*args, **kwargs):
    class Result:
        returncode = 0
        stderr = ""

    return Result()


def _frame_fn_factory(calls):
    def frame_fn(i, idx):
        calls.append(idx)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, idx])
        return fig

    return frame_fn


def test_frame_directory_created(tmp_path):
    img_dir = tmp_path / "frames"
    assert not img_dir.exists()

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "subprocess.run", side_effect=_fake_ffmpeg_run
    ):
        render_movie(img_dir, [1, 2], _frame_fn_factory([]), fps=5)

    assert img_dir.exists()
    assert (img_dir / "fig_t-000.png").exists()
    assert (img_dir / "fig_t-001.png").exists()


def test_skip_existing_frames_by_default(tmp_path):
    img_dir = tmp_path / "frames"
    calls = []

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "subprocess.run", side_effect=_fake_ffmpeg_run
    ):
        render_movie(img_dir, [1, 2, 3], _frame_fn_factory(calls), fps=5)
        assert calls == [1, 2, 3]

        calls.clear()
        render_movie(img_dir, [1, 2, 3], _frame_fn_factory(calls), fps=5, rerun_all=False)
        assert calls == []  # all frames already exist, none recomputed


def test_rerun_all_recomputes_every_frame(tmp_path):
    img_dir = tmp_path / "frames"
    calls = []

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "subprocess.run", side_effect=_fake_ffmpeg_run
    ):
        render_movie(img_dir, [1, 2, 3], _frame_fn_factory(calls), fps=5)
        calls.clear()
        render_movie(img_dir, [1, 2, 3], _frame_fn_factory(calls), fps=5, rerun_all=True)

    assert calls == [1, 2, 3]


def test_on_error_continue_skips_bad_frame(tmp_path):
    img_dir = tmp_path / "frames"

    def frame_fn(i, idx):
        if idx == "bad":
            raise ValueError("boom")
        fig, ax = plt.subplots()
        return fig

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "subprocess.run", side_effect=_fake_ffmpeg_run
    ):
        render_movie(img_dir, [1, "bad", 2], frame_fn, fps=5, on_error="continue")

    assert (img_dir / "fig_t-000.png").exists()
    assert not (img_dir / "fig_t-001.png").exists()
    assert (img_dir / "fig_t-002.png").exists()


def test_on_error_raise_propagates(tmp_path):
    img_dir = tmp_path / "frames"

    def frame_fn(i, idx):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        render_movie(img_dir, [1], frame_fn, fps=5, on_error="raise")


def test_frame_returning_none_is_skipped(tmp_path):
    img_dir = tmp_path / "frames"

    def frame_fn(i, idx):
        return None

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "subprocess.run", side_effect=_fake_ffmpeg_run
    ):
        render_movie(img_dir, [1, 2], frame_fn, fps=5)

    assert list(img_dir.glob("*.png")) == []


def test_figures_closed_after_each_frame(tmp_path):
    img_dir = tmp_path / "frames"
    plt.close("all")

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "subprocess.run", side_effect=_fake_ffmpeg_run
    ):
        render_movie(img_dir, [1, 2, 3], _frame_fn_factory([]), fps=5)

    assert plt.get_fignums() == []


def test_ffmpeg_not_found_raises_clear_error(tmp_path):
    img_dir = tmp_path / "frames"

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "subprocess.run", side_effect=_fake_ffmpeg_run
    ):
        render_movie(img_dir, [1], _frame_fn_factory([]), fps=5)

    with patch("shutil.which", return_value=None):
        with pytest.raises(FFmpegNotFoundError):
            ffmpeg_frames_to_video(img_dir, "*.png", img_dir / "out.mp4")


def test_ffmpeg_nonzero_exit_raises(tmp_path):
    img_dir = tmp_path / "frames"
    img_dir.mkdir()
    fig, ax = plt.subplots()
    fig.savefig(img_dir / "fig_t-000.png")
    plt.close(fig)

    def failing_run(*args, **kwargs):
        class Result:
            returncode = 1
            stderr = "ffmpeg exploded"

        return Result()

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "subprocess.run", side_effect=failing_run
    ):
        with pytest.raises(RuntimeError, match="ffmpeg exploded"):
            ffmpeg_frames_to_video(img_dir, "*.png", img_dir / "out.mp4")


def test_ffmpeg_invoked_with_expected_args(tmp_path):
    img_dir = tmp_path / "frames"
    img_dir.mkdir()
    fig, ax = plt.subplots()
    fig.savefig(img_dir / "fig_t-000.png")
    plt.close(fig)

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "subprocess.run", side_effect=_fake_ffmpeg_run
    ) as mock_run:
        ffmpeg_frames_to_video(img_dir, "*.png", img_dir / "out.mp4", fps=15)

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "-y" in cmd
    assert "15" in cmd
    assert str(img_dir / "out.mp4") == cmd[-1]
