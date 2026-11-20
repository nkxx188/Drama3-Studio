"""Reference audio import that also accepts the audio track of video files.

Only the audio stream is kept: video containers are decoded once with FFmpeg
into a float WAV, so every later step (waveform, ASR, codec preparation) works
on a plain WAV file regardless of the original container.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
AUDIO_SUFFIXES = frozenset({
    ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".wma", ".aif", ".aiff",
})
VIDEO_SUFFIXES = frozenset({
    ".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".wmv", ".flv",
    ".ts", ".mpg", ".mpeg", ".3gp",
})
MAX_SECONDS = 120
UNSUPPORTED = "Choose a WAV, MP3, FLAC, OGG, M4A or MP4 audio file"
NO_FFMPEG = "Importing video audio needs FFmpeg. Install FFmpeg or add ffmpeg.exe to PATH."
NO_AUDIO = "This video has no readable audio track."
UNREADABLE = "Could not read audio from this video."
TOO_LONG = "Video audio must be 120 seconds or shorter."


def suffix_of(filename):
    return Path(str(filename or "")).suffix.lower()


def is_video(filename):
    return suffix_of(filename) in VIDEO_SUFFIXES


def ffmpeg_path(root=ROOT):
    override = os.environ.get("DRAMA_FFMPEG", "").strip()
    candidates = [Path(override)] if override else []
    root = Path(root)
    candidates += [
        root / "runtime" / "ffmpeg" / "bin" / "ffmpeg.exe",
        root / "runtime" / "ffmpeg" / "bin" / "ffmpeg",
        root / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
        root / "vendor" / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffmpeg")


def prepare_reference_audio(data, filename, root=ROOT):
    """Return ``(audio_bytes, stored_suffix)`` for an uploaded reference."""
    suffix = suffix_of(filename)
    if suffix in VIDEO_SUFFIXES:
        return extract_video_audio(data, suffix, root), ".wav"
    if suffix in AUDIO_SUFFIXES:
        return data, suffix
    raise ValueError(UNSUPPORTED)


def extract_video_audio(data, suffix=".mp4", root=ROOT):
    """Decode the first audio stream of a video container into float WAV bytes."""
    ffmpeg = ffmpeg_path(root)
    if not ffmpeg:
        raise ValueError(NO_FFMPEG)
    work = Path(root) / ".cache" / "tmp"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work, prefix="video-audio-") as folder:
        source = Path(folder) / ("input" + (suffix or ".mp4"))
        target = Path(folder) / "audio.wav"
        source.write_bytes(data)
        command = [
            ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(source),
            "-map", "0:a:0", "-vn", "-t", f"{MAX_SECONDS + 0.5:g}",
            "-c:a", "pcm_f32le", str(target),
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=180, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ValueError(UNREADABLE) from error
        stderr = result.stderr or ""
        if result.returncode or not target.is_file():
            if "matches no streams" in stderr or "does not contain any stream" in stderr:
                raise ValueError(NO_AUDIO)
            raise ValueError(UNREADABLE)
        import soundfile as sf
        try:
            info = sf.info(str(target))
        except (RuntimeError, sf.LibsndfileError) as error:
            raise ValueError(UNREADABLE) from error
        if not 0 < info.duration:
            raise ValueError(NO_AUDIO)
        if info.duration > MAX_SECONDS:
            raise ValueError(TOO_LONG)
        return target.read_bytes()
