"""Download or resume the pinned official Drama 3 checkpoint."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drama_low_vram.model_download import main


if __name__ == "__main__":
    main()
