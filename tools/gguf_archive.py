"""Compatibility import for existing conversion tools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drama_low_vram.gguf_archive import (  # noqa: E402, F401
    Archive, BLOCKS, FORMATS, INTEREST, Tensor, pack_string, read_number,
    read_string, read_value, scalar_metadata, write_header,
)
