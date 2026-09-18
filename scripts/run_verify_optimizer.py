#!/usr/bin/env python3
"""Execute and retain the Phase 5 scientific gates; failures remain failures."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from pathlib import Path
import sys
import subprocess
import shutil
import json
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scripts.phase5_records import environment, source_hashes, write_json
from scripts.phase5_experiments import (
    audit,
    ill_conditioned_optimization_sweep,
    nonconvex_stochastic_benchmarks,
    ards_monotonicity_and_asymptotics_study,
    architectural_isolation_audit,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/phase5')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    r = {
        'phase': 5,
        'gate_version': 1,
        'environment': environment(),
        'command': sys.argv,
    }

    # Inherited Phase 1, 2, 3, and 4 validation
    p1_metrics_path = ROOT / 'results/phase1/metrics.json'
    p2_metrics_path = ROOT / 'results/phase2/metrics.json'
    p3_metrics_path = ROOT / 'results/phase3/metrics.json'
    p4_metrics_path = ROOT / 'results/phase4/metrics.json'
    p1_pass_path = ROOT / 'results/phase1/PASS.md'
    p2_pass_path = ROOT / 'results/phase2/PASS.md'
    p3_pass_path = ROOT / 'results/phase3/PASS.md'
    p4_pass_path = ROOT / 'results/phase4/PASS.md'

    r['inherited_phases'] = {
        'phase1_metrics_exist': p1_metrics_path.exists(),
        'phase1_pass_exists': p1_pass_path.exists(),
        'phase2_metrics_exist': p2_metrics_path.exists(),
        'phase2_pass_exists': p2_pass_path.exists(),
        'phase3_metrics_exist': p3_metrics_path.exists(),
        'phase3_pass_exists': p3_pass_path.exists(),
        'phase4_metrics_exist': p4_metrics_path.exists(),
        'phase4_pass_exists': p4_pass_path.exists(),
    }
    r['inherited_phases']['passed'] = all(r['inherited_phases'].values())

    # Formal Lean verification
    lake = shutil.which('lake') or str(Path.home() / '.elan/bin/lake')
    build = subprocess.run([lake, 'build'], cwd=ROOT / 'formal', capture_output=True, text=True)
    (out / 'lean-build.log').write_text(build.stdout + build.stderr)
    violations = [
        str(f.relative_to(ROOT))
        for f in (ROOT / 'formal').rglob('*.lean')
        if '.lake' not in f.parts and re.search(r'\b(sorry|admit|axiom)\b', f.read_text())
    ]
    r['formal'] = {
        'passed': build.returncode == 0 and not re.search('warning', build.stdout + build.stderr, re.I) and not violations,
        'proof_scan_violations': violations,
    }

    # Full unit test suite
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, capture_output=True, text=True)
    (out / 'pytest.log').write_text(test.stdout + test.stderr)
    r['unit_tests'] = {'passed': test.returncode == 0}

    # Zero-transcendental AST and token audit
    r['purity'] = audit()

    # Empirical scientific studies
    r['ill_conditioned_sweep'] = ill_conditioned_optimization_sweep()
    r['nonconvex_benchmarks'] = nonconvex_stochastic_benchmarks()
    r['ards_properties'] = ards_monotonicity_and_asymptotics_study()
    r['architectural_isolation'] = architectural_isolation_audit()

    # Hardware evidence check
    tpu = out / 'tpu/metrics.json'
    r['hardware'] = {'passed': False, 'reason': 'Missing TPU evidence'}
    if tpu.exists():
        h = json.loads(tpu.read_text())
        matched = h.get('environment', {}).get('source_sha256') == r['environment']['source_sha256']
        inventory = (
            h.get('device_count') == 16
            and h.get('process_count') == 4
            and len(h.get('devices', [])) == 16
            and all('TPU v4' in d.get('kind', '') for d in h.get('devices', []))
        )
        inventory = (
            inventory
            and len(h.get('parity', {}).get('rows', [])) >= 8
            and all(v['passed'] for v in h['parity']['rows'])
        )
        inventory = (
            inventory
            and len(h.get('benchmarks', {}).get('rows', [])) >= 4
            and all(
                v['repetitions'] >= 100
                and all(g['ratio'] >= 0.90 for g in v['gates'].values())
                for v in h['benchmarks']['rows']
            )
        )
        inventory = inventory and all(
            h.get(k, {}).get('passed')
            for k in (
                'parity',
                'benchmarks',
                'hlo_audit',
            )
        )
        inventory = (
            inventory
            and len(h.get('hlo_audit', {}).get('rows', [])) >= 2
            and all(
                (not row.get('name', '').endswith('ards_step'))
                or (row.get('raw_sqrt_count') == 0 and row.get('rsqrt_count', 0) >= 2)
                for row in h.get('hlo_audit', {}).get('rows', [])
            )
        )
        r['hardware'] = {
            'passed': bool(matched and inventory and h.get('passed')),
            'source_matches': matched,
            'inventory_matches': inventory,
            'path': str(tpu),
        }

    gates = (
        'inherited_phases',
        'formal',
        'unit_tests',
        'purity',
        'ill_conditioned_sweep',
        'nonconvex_benchmarks',
        'ards_properties',
        'architectural_isolation',
        'hardware',
    )
    r['passed'] = all(r[k]['passed'] for k in gates)
    if source_hashes() != r['environment']['source_sha256']:
        r['passed'] = False
        r['source_changed'] = True
    r['status'] = 'PASS' if r['passed'] else 'FAIL'

    write_json(out / 'metrics.json', r)
    print(r['status'], flush=True)
    return 0 if r['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
