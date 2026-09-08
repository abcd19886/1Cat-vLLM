# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export H3 video and its synchronous 32 kHz audio."""

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf


def export_video(video, audio, output_dir, *, fps=24):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Pipeline video is B,T,H,W,C, audio is B,C,samples.
    video = video.detach().cpu()
    audio = audio.detach().float().cpu()
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[-1] != 3:
        raise ValueError(f"unexpected H3 video shape {tuple(video.shape)}")
    if audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] not in (1, 2):
        raise ValueError(f"unexpected H3 audio shape {tuple(audio.shape)}")
    if not np.isfinite(audio.numpy()).all():
        raise ValueError("H3 produced non-finite audio")
    waveform = audio[0].numpy().T
    wav = root / "audio.wav"
    sf.write(wav, waveform, 32000, subtype="FLOAT")
    _, frames, height, width, _ = video.shape
    output = root / "video.mp4"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(wav),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    with (root / "encode.log").open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            assert process.stdin is not None
            for frame in video[0]:
                process.stdin.write(frame.contiguous().numpy().tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(
                    f"ffmpeg video export failed; see {root / 'encode.log'}"
                )
        except BaseException:
            process.kill()
            process.wait()
            raise
    return {
        "video": str(output),
        "audio": str(wav),
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
    }
