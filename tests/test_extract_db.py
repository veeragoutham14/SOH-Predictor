from pathlib import Path

from src.extract_db import _next_part_index


def test_next_part_index_continues_after_highest_existing_file(tmp_path: Path) -> None:
    assert _next_part_index(tmp_path) == 0

    (tmp_path / "part-00000.parquet").touch()
    (tmp_path / "part-00020.parquet").touch()
    (tmp_path / "part-invalid.parquet").touch()
    (tmp_path / "different-99999.parquet").touch()

    assert _next_part_index(tmp_path) == 21
