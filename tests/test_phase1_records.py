import json

from scripts.run_verify_primitives import hardware_evidence


def test_missing_hardware_cannot_pass(tmp_path):
    assert not hardware_evidence(tmp_path/"missing.json",{})["passed"]


def test_stale_hardware_cannot_pass(tmp_path):
    path=tmp_path/"metrics.json"
    path.write_text(json.dumps({"passed":True,"environment":{"source_sha256":{"src/primitives.py":"old"}}}))
    assert not hardware_evidence(path,{"src/primitives.py":"new"})["passed"]


def test_incomplete_inventory_cannot_pass(tmp_path):
    path=tmp_path/"metrics.json"
    path.write_text(json.dumps({"passed":True,"environment":{"source_sha256":{}},"device_count":16,"process_count":4,
                               "devices":[{"kind":"TPU v4"}]*16}))
    assert not hardware_evidence(path,{})["passed"]
