from __future__ import annotations

from collections.abc import Sequence

from src.extract_db import main as extract_main


def main(argv: Sequence[str] | None = None) -> int:
    """Project entry point for the current extraction phase."""
    return extract_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
