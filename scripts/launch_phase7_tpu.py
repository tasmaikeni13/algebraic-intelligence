#!/usr/bin/env python3
"""Copy a committed snapshot and run Phase 7 pilot pretraining on the idle TPU slice.

Does not stop other jobs or create cloud resources. Source snapshots are isolated
in new per-run directories on each host. Requires gcloud and existing SSH access.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="my-tpu-v4")
    parser.add_argument("--zone", default="us-central2-b")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase7/tpu")
    parser.add_argument("--steps", type=int, default=100_000, help="Total pretraining steps")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=10_000)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def run(command, log):
        with (out / log).open("w") as file:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                file.write(line)
                file.flush()
                print(line, end="", flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"Command failed with exit {code}; see {out / log}")

    def ssh(command, log, worker="all"):
        run(["gcloud", "compute", "tpus", "tpu-vm", "ssh", args.name, "--zone", args.zone,
             "--worker", worker, "--quiet", "--command", command], log)

    ssh("if sudo -n fuser -s /dev/accel*; then echo 'TPU devices are in use; leave the existing job running.'; exit 73; fi", "availability.log")

    # Ensure the bundle contains reviewed source files.
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts", "tests", "formal", "phases/phase7.md", "requirements.txt", "requirements-tpu.txt"], cwd=ROOT, check=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "src", "scripts", "tests", "formal", "requirements.txt", "requirements-tpu.txt"], cwd=ROOT, text=True)
    if untracked:
        raise RuntimeError(f"Commit Phase 7 source files before launching a distributed snapshot:\n{untracked}")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if not branch:
        raise RuntimeError("Launch from a named branch.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"algebraic-phase7-{commit[:12]}-{stamp}"
    remote = f"/tmp/{label}"
    quote = shlex.quote

    with tempfile.TemporaryDirectory(prefix="algebraic-phase7-") as temporary:
        bundle = Path(temporary) / "source.bundle"
        subprocess.run(["git", "bundle", "create", str(bundle), branch], cwd=ROOT, check=True)
        run(["gcloud", "compute", "tpus", "tpu-vm", "scp", str(bundle), f"{args.name}:{remote}.bundle", "--zone", args.zone, "--worker", "all", "--quiet"], "copy.log")

    setup = (f"git clone --quiet --branch {quote(branch)} {quote(remote+'.bundle')} {quote(remote)} && "
             f"python3 -m venv {quote(remote+'/venv')} && "
             f"{quote(remote+'/venv/bin/pip')} install -r {quote(remote+'/requirements-tpu.txt')}")
    ssh(setup, "setup.log")

    # Ensure tokenized dataset is available on all workers (copy from host 0 cache if available)
    if (ROOT / "data/train.npy").exists() and (ROOT / "data/valid.npy").exists():
        print("Distributing pre-tokenized dataset cache to all TPU workers...", flush=True)
        import shutil
        dest_0 = Path(remote) / "data"
        dest_0.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "data/train.npy", dest_0 / "train.npy")
        shutil.copy2(ROOT / "data/valid.npy", dest_0 / "valid.npy")
        for w in range(1, 4):
            ssh(f"mkdir -p {quote(remote+'/data')}", f"mkdir-data-{w}.log", worker=str(w))
            run(["gcloud", "compute", "tpus", "tpu-vm", "scp", str(ROOT / "data/train.npy"), str(ROOT / "data/valid.npy"),
                 f"{args.name}:{remote}/data/", "--zone", args.zone, "--worker", str(w), "--quiet"], f"scp-data-{w}.log")

    # Separate CPU environments on every host do not initialize the TPU runtime.
    ssh(f"cd {quote(remote)} && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 venv/bin/python -m pytest -q", "host-tests.log")
    (out / "snapshot.txt").write_text(f"commit={commit}\nremote_directory={remote}\n")

    cmd = (
        f"cd {quote(remote)} && PYTHONUNBUFFERED=1 venv/bin/python scripts/run_pilot_15m.py "
        f"--output {quote(remote+'/measurements')} --steps {args.steps} "
        f"--log-every {args.log_every} --eval-every {args.eval_every}"
    )

    try:
        ssh(cmd, "run.log")
    finally:
        download = out / "download" / label
        download.mkdir(parents=True)
        for worker in range(4):
            run(["gcloud", "compute", "tpus", "tpu-vm", "scp", "--recurse",
                 f"{args.name}:{remote}/measurements", str(download / f"worker-{worker}"),
                 "--zone", args.zone, "--worker", str(worker), "--quiet"], f"download-{worker}.log")
        candidates = list(download.rglob("metrics.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one coordinator record; found {len(candidates)} in {download}")
        import shutil
        for file in candidates[0].parent.iterdir():
            if file.is_file():
                shutil.copy2(file, out / file.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
