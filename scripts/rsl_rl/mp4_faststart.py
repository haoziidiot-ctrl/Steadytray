import argparse
import os
import struct
from dataclasses import dataclass
from pathlib import Path


CONTAINER_BOX_TYPES = {
    "moov",
    "trak",
    "mdia",
    "minf",
    "stbl",
    "edts",
    "dinf",
    "mvex",
    "moof",
    "traf",
    "mfra",
    "udta",
}


@dataclass(frozen=True)
class Mp4Box:
    start: int
    size: int
    header_size: int
    type: str

    @property
    def end(self) -> int:
        return self.start + self.size


def _read_u32(buf: bytes, offset: int) -> int:
    return struct.unpack_from(">I", buf, offset)[0]


def _read_u64(buf: bytes, offset: int) -> int:
    return struct.unpack_from(">Q", buf, offset)[0]


def _write_u32(buf: bytearray, offset: int, value: int) -> None:
    struct.pack_into(">I", buf, offset, value)


def _write_u64(buf: bytearray, offset: int, value: int) -> None:
    struct.pack_into(">Q", buf, offset, value)


def _iter_boxes(buf: bytes, start: int = 0, end: int | None = None):
    if end is None:
        end = len(buf)
    offset = start
    while offset + 8 <= end:
        size = _read_u32(buf, offset)
        box_type = buf[offset + 4 : offset + 8].decode("latin1")
        header_size = 8
        if size == 1:
            if offset + 16 > end:
                break
            size = _read_u64(buf, offset + 8)
            header_size = 16
        elif size == 0:
            size = end - offset
        if size < header_size or offset + size > end:
            break
        yield Mp4Box(start=offset, size=size, header_size=header_size, type=box_type)
        offset += size


def _patch_chunk_offsets_in_container(buf: bytearray, container: Mp4Box, delta: int) -> None:
    payload_start = container.start + container.header_size
    if container.type == "meta":
        payload_start += 4
    for box in _iter_boxes(buf, payload_start, container.end):
        if box.type == "stco":
            entry_count = _read_u32(buf, box.start + box.header_size + 4)
            table_offset = box.start + box.header_size + 8
            for i in range(entry_count):
                entry_offset = table_offset + i * 4
                _write_u32(buf, entry_offset, _read_u32(buf, entry_offset) + delta)
        elif box.type == "co64":
            entry_count = _read_u32(buf, box.start + box.header_size + 4)
            table_offset = box.start + box.header_size + 8
            for i in range(entry_count):
                entry_offset = table_offset + i * 8
                _write_u64(buf, entry_offset, _read_u64(buf, entry_offset) + delta)
        elif box.type in CONTAINER_BOX_TYPES:
            _patch_chunk_offsets_in_container(buf, box, delta)


def needs_faststart(path: str | os.PathLike[str]) -> bool:
    data = Path(path).read_bytes()
    top_level = list(_iter_boxes(data))
    moov = next((box for box in top_level if box.type == "moov"), None)
    mdat = next((box for box in top_level if box.type == "mdat"), None)
    if moov is None or mdat is None:
        raise ValueError(f"Missing moov/mdat in {path}")
    return moov.start > mdat.start


def rewrite_faststart(src: str | os.PathLike[str], dst: str | os.PathLike[str] | None = None) -> Path:
    src_path = Path(src)
    dst_path = Path(dst) if dst is not None else src_path
    data = src_path.read_bytes()
    top_level = list(_iter_boxes(data))

    moov = next((box for box in top_level if box.type == "moov"), None)
    mdat = next((box for box in top_level if box.type == "mdat"), None)
    if moov is None or mdat is None:
        raise ValueError(f"Missing moov/mdat in {src_path}")
    if moov.start < mdat.start:
        if dst_path != src_path:
            dst_path.write_bytes(data)
        return dst_path

    moov_bytes = bytearray(data[moov.start : moov.end])
    _patch_chunk_offsets_in_container(moov_bytes, Mp4Box(0, len(moov_bytes), moov.header_size, moov.type), moov.size)

    reordered = bytearray()
    inserted_moov = False
    for box in top_level:
        if box.type == "moov":
            continue
        if not inserted_moov and box.type == "mdat":
            reordered.extend(moov_bytes)
            inserted_moov = True
        reordered.extend(data[box.start : box.end])
    if not inserted_moov:
        raise ValueError(f"Failed to place moov before mdat for {src_path}")

    if dst_path == src_path:
        tmp_path = src_path.with_suffix(src_path.suffix + ".faststart.tmp")
        tmp_path.write_bytes(reordered)
        os.replace(tmp_path, src_path)
    else:
        dst_path.write_bytes(reordered)
    return dst_path


def faststart_sibling_path(path: str | os.PathLike[str]) -> Path:
    src = Path(path)
    return src.with_name(f"{src.stem}.faststart{src.suffix}")


def rewrite_faststart_in_place_if_needed(path: str | os.PathLike[str]) -> bool:
    if not needs_faststart(path):
        return False
    rewrite_faststart(path, path)
    return True


def rewrite_faststart_folder_in_place(folder: str | os.PathLike[str]) -> list[Path]:
    folder_path = Path(folder)
    rewritten = []
    for mp4_path in sorted(folder_path.glob("*.mp4")):
        if rewrite_faststart_in_place_if_needed(mp4_path):
            rewritten.append(mp4_path)
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite MP4 files so that moov is placed before mdat.")
    parser.add_argument("input", help="Input MP4 path.")
    parser.add_argument("--output", help="Optional output path. Defaults to <input>.faststart.mp4.")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Rewrite the input file in place instead of creating a sibling faststart file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = input_path if args.inplace else Path(args.output) if args.output else faststart_sibling_path(input_path)
    changed = needs_faststart(input_path)
    result_path = rewrite_faststart(input_path, output_path)
    status = "rewritten" if changed else "already-faststart"
    print(f"{status}: {result_path}")


if __name__ == "__main__":
    main()
