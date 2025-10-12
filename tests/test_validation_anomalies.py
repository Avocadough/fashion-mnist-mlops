import json
from pathlib import Path

def test_validation_reports_and_anomaly_fields():
    # ต้องมีไฟล์รายงานจาก step 1
    report = Path("mlruns")
    # หาไฟล์ reports/validation.json ล่าสุดจาก mlruns (หยาบแบบสแกน)
    validation_jsons = list(Path(".").glob("mlruns/**/artifacts/reports/validation.json"))
    assert validation_jsons, "validation.json not found; run step 1"
    # ใช้อันหลังสุด
    vfile = sorted(validation_jsons)[-1]
    data = json.loads(vfile.read_text(encoding="utf-8"))
    # ฟิลด์ที่ต้องมี
    for key in ["dataset", "shape", "nulls", "label_unique", "anomalies"]:
        assert key in data
    # anomalies ต้องมีตัวชี้วัดตามที่กำหนด
    for k in ["pixel_out_of_range_low", "pixel_out_of_range_high", "duplicate_rows", "zscore_outlier_cells_gt6_in_sample1k"]:
        assert k in data["anomalies"]
