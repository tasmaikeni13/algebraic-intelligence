"""Tests for Phase 6 empirical records and hardware evidence validation."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _check_hardware(path, expected_hashes):
    if not path.exists():
        return {"passed": False, "reason": "Missing TPU evidence"}
    h = json.loads(path.read_text())
    matched = h.get("environment", {}).get("source_sha256") == expected_hashes
    inventory = (
        h.get("device_count") == 16
        and h.get("process_count") == 4
        and len(h.get("devices", [])) == 16
        and all("TPU v4" in d.get("kind", "") for d in h.get("devices", []))
    )
    inventory = (
        inventory
        and len(h.get("parity", {}).get("rows", [])) >= 4
        and all(v["passed"] for v in h["parity"]["rows"])
    )
    inventory = (
        inventory
        and len(h.get("benchmarks", {}).get("rows", [])) >= 2
        and all(
            v["repetitions"] >= 50
            and v.get("afa_tflops", 0.0) >= 0.80 * v.get("baseline_tflops", 0.0)
            for v in h["benchmarks"]["rows"]
        )
    )
    inventory = (
        inventory
        and h.get("bandwidth", {}).get("passed", False)
        and h.get("bandwidth", {}).get("sustained_gb_s", 0.0) >= 840.0
    )
    inventory = (
        inventory
        and h.get("ring_attention", {}).get("passed", False)
        and h.get("ring_attention", {}).get("rel_error", 1.0) <= 1.0e-6
    )
    inventory = (
        inventory
        and h.get("hlo_audit", {}).get("passed", False)
        and h.get("hlo_audit", {}).get("transcendental_opcodes_count", 1) == 0
    )
    inventory = inventory and all(
        h.get(k, {}).get("passed")
        for k in (
            "parity",
            "benchmarks",
            "bandwidth",
            "ring_attention",
            "hlo_audit",
        )
    )
    return {
        "passed": bool(matched and inventory and h.get("passed")),
        "source_matches": matched,
        "inventory_matches": inventory,
        "path": str(path),
    }


def test_missing_tpu_hardware_cannot_pass(tmp_path):
    assert not _check_hardware(tmp_path / "missing.json", {})["passed"]


def test_stale_tpu_hardware_cannot_pass(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"passed": True, "environment": {"source_sha256": {"src/kernels/pallas_afa.py": "old"}}}))
    assert not _check_hardware(path, {"src/kernels/pallas_afa.py": "new"})["passed"]


def test_incomplete_tpu_inventory_cannot_pass(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps({
            "passed": True,
            "environment": {"source_sha256": {}},
            "device_count": 8,  # Only 8 chips instead of 16
            "process_count": 2,
            "devices": [{"kind": "TPU v4"}] * 8,
        })
    )
    assert not _check_hardware(path, {})["passed"]


def test_committed_tpu_metrics_validates_cleanly():
    tpu_path = ROOT / "results/phase6/tpu/metrics.json"
    if not tpu_path.exists():
        pytest.skip("TPU evidence not yet produced")
    h = json.loads(tpu_path.read_text())
    hashes = h["environment"]["source_sha256"]
    res = _check_hardware(tpu_path, hashes)
    assert res["passed"] is True, f"Committed TPU evidence failed validation: {res}"
    assert res["inventory_matches"] is True
    assert res["source_matches"] is True
