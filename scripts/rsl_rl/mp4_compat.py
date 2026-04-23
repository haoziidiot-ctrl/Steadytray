import argparse
import os
import subprocess
from pathlib import Path


def _get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio_ffmpeg is required to generate compatibility MP4 files.") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def compat_sibling_path(path: str | os.PathLike[str]) -> Path:
    src = Path(path)
    return src.with_name(f"{src.stem}.compat{src.suffix}")


def rewrite_compat(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str] | None = None,
    *,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    src_path = Path(src)
    dst_path = Path(dst) if dst is not None else src_path
    ffmpeg_exe = _get_ffmpeg_exe()

    tmp_path = dst_path.with_suffix(dst_path.suffix + ".compat.tmp") if dst_path == src_path else dst_path
    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "main",
        "-level",
        "4.0",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown ffmpeg failure"
        raise RuntimeError(f"ffmpeg compatibility rewrite failed for {src_path}: {stderr}")
    if tmp_path != dst_path:
        os.replace(tmp_path, dst_path)
    return dst_path


def rewrite_compat_folder(folder: str | os.PathLike[str]) -> list[Path]:
    folder_path = Path(folder)
    rewritten = []
    for mp4_path in sorted(folder_path.glob("*.mp4")):
        if mp4_path.stem.endswith(".compat"):
            continue
        rewritten.append(rewrite_compat(mp4_path, mp4_path))
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate broadly compatible H.264/yuv420p MP4 files using the bundled ffmpeg."
    )
    parser.add_argument("input", help="Input MP4 path.")
    parser.add_argument("--output", help="Optional output path. Defaults to rewriting the input in place.")
    parser.add_argument("--crf", type=int, default=18, help="x264 CRF value. Lower means higher quality.")
    parser.add_argument("--preset", default="medium", help="x264 preset passed to ffmpeg.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    result_path = rewrite_compat(input_path, output_path, crf=args.crf, preset=args.preset)
    print(result_path)


if __name__ == "__main__":
    main()
