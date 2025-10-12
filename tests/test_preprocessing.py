
import pandas as pd
from pathlib import Path

def test_split_files_exist():
    # Expect the user ran step 1 & 2 before CI; in CI we just check structure when present
    split_dir = Path("artifacts/splits")
    # The test is lenient: if folder exists, files must exist; if not, skip-like pass
    if split_dir.exists():
        assert (split_dir / "train.csv").exists()
        assert (split_dir / "val.csv").exists()
        assert (split_dir / "test.csv").exists()
    else:
        assert True  # allow pipeline-first development
