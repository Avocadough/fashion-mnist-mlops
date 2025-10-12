# tests/test_validation_anomalies.py
from pathlib import Path
import json

def test_validation_report_exists_and_has_anomalies():
    # หาไฟล์รายงานล่าสุดจาก mlruns/**/artifacts/reports/validation.json
    files = sorted(Path(".").glob("mlruns/**/artifacts/reports/validation.json"))
    assert files, "validation.json not found. Did you run step 1?"
    data = json.loads(files[-1].read_text(encoding="utf-8"))

    # ฟิลด์หลักที่คาดหวัง
    for k in ["dataset", "shape", "nulls", "label_unique", "anomalies"]:
        assert k in data, f"missing field: {k}"

    a = data["anomalies"]
    for k in [
        "pixel_out_of_range_low",
        "pixel_out_of_range_high",
        "duplicate_rows",
        "zscore_outlier_cells_gt6_in_sample1k",
    ]:
        assert k in a, f"missing anomaly metric: {k}"
