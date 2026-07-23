#!/usr/bin/env python3
"""
setup_wizard.py -- interactive configuration + validation for Symplify.

Goal: catch cluster-specific problems (bad partition, missing account, wrong
env names, missing paths) BEFORE a user launches a real job -- instead of a
cryptic "sbatch returned non-zero exit status 1" mid-run.

Usage:
    python setup_wizard.py           # interactive setup: ask -> validate -> write config.yaml
    python setup_wizard.py --check   # validate the existing config.yaml only (no prompts)

Wire into run.py (optional, two lines near the top of main):
    if "--setup" in sys.argv:
        from setup_wizard import run_setup; run_setup(); sys.exit()
    if "--check" in sys.argv:
        from setup_wizard import check_only; sys.exit(0 if check_only() else 1)

Only depends on PyYAML (already used by the server) + the standard library.
"""

import os
import re
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"

# --------------------------------------------------------------------------- #
#  Pretty output
# --------------------------------------------------------------------------- #
_TTY = sys.stdout.isatty()


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def head(s):
    print("\n" + _c("36;1", s))


def ok(s):
    print("  " + _c("32", "[ok]  ") + s)


def warn(s):
    print("  " + _c("33", "[warn]") + " " + s)


def bad(s):
    print("  " + _c("31", "[fail]") + " " + s)


def ask(prompt, default=""):
    hint = f" [{default}]" if default else ""
    try:
        r = input(f"  {prompt}{hint}: ").strip()
    except EOFError:
        r = ""
    return r or default


def ask_yes(prompt, default=True):
    d = "Y/n" if default else "y/N"
    r = ask(f"{prompt} ({d})", "")
    if not r:
        return default
    return r.lower().startswith("y")


# --------------------------------------------------------------------------- #
#  Config load / save  (save preserves comments by editing lines in place)
# --------------------------------------------------------------------------- #
def load_config():
    if yaml is None:
        bad("PyYAML is not installed in this environment (need it to read config.yaml).")
        return None
    if not CONFIG.exists():
        bad(f"config.yaml not found at {CONFIG}")
        return None
    with open(CONFIG) as f:
        return yaml.safe_load(f) or {}


def write_slurm_settings(settings):
    """Replace gpu_partition / cpu_partition / account / extra_flags *inside the
    slurm: block only*, preserving all comments and the rest of the file.
    (extra_flags also appears under pbs:/sge:, so we must scope to slurm:.)"""
    lines = CONFIG.read_text().splitlines()
    out, in_slurm, slurm_indent, done = [], False, None, set()
    keys = ("gpu_partition", "cpu_partition", "account", "extra_flags")
    for line in lines:
        m = re.match(r"^(\s*)slurm:\s*$", line)
        if m:
            in_slurm, slurm_indent = True, len(m.group(1))
            out.append(line)
            continue
        if in_slurm:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            # a real key at or above the slurm indent level ends the block
            if stripped and not stripped.startswith("#") and indent <= slurm_indent:
                in_slurm = False
            else:
                km = re.match(r"^(\s*)(\w+):\s*(.*)$", line)
                if km and km.group(2) in keys and km.group(2) in settings and km.group(2) not in done:
                    indent_s, key, rest = km.group(1), km.group(2), km.group(3)
                    cm = re.search(r"(\s+#.*)$", rest)
                    comment = cm.group(1) if cm else ""
                    out.append(f'{indent_s}{key}: "{settings[key]}"{comment}')
                    done.add(key)
                    continue
        out.append(line)
    CONFIG.write_text("\n".join(out) + "\n")
    return done


# --------------------------------------------------------------------------- #
#  SLURM live probe -- the important part
# --------------------------------------------------------------------------- #
def _build_probe(gpu_partition, account, extra_flags, want_gpu=True):
    h = [
        "#!/bin/bash",
        "#SBATCH --job-name=symplify_probe",
        "#SBATCH --time=00:01:00",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=1G",
        "#SBATCH --output=/dev/null",
        "#SBATCH --error=/dev/null",
    ]
    if gpu_partition:
        h.append(f"#SBATCH --partition={gpu_partition}")
    if want_gpu:
        h.append("#SBATCH --gres=gpu:1")
    if account:
        h.append(f"#SBATCH --account={account}")
    if extra_flags:
        h.append(f"#SBATCH {extra_flags}")
    h.append("echo probe")
    return "\n".join(h) + "\n"


def probe_slurm(gpu_partition, account, extra_flags, want_gpu=True):
    """Submit a 1-minute dummy job with the given settings, read sbatch's
    result, cancel it if accepted. Returns (ok: bool, message: str, hint: str).
    `hint` is one of: '', 'blank_gpu_partition', 'account', 'partition', 'time'."""
    if not shutil.which("sbatch"):
        return False, "sbatch not found on PATH -- are you on a login/compute node with SLURM?", ""
    script = _build_probe(gpu_partition, account, extra_flags, want_gpu)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False)
    tmp.write(script)
    tmp.close()
    try:
        r = subprocess.run(["sbatch", tmp.name], capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        os.unlink(tmp.name)
        return False, f"could not run sbatch: {e}", ""
    os.unlink(tmp.name)

    if r.returncode == 0:
        # "Submitted batch job 12345" -> cancel it so it never runs
        m = re.search(r"(\d+)", r.stdout)
        if m and shutil.which("scancel"):
            subprocess.run(["scancel", m.group(1)], capture_output=True, text=True)
        return True, "sbatch accepted the job (test job cancelled).", ""

    err = (r.stderr + r.stdout).lower()
    if "not allowed" in err and "partition" in err:
        return False, r.stderr.strip(), "blank_gpu_partition"
    if "invalid partition" in err or "partition" in err and "specified" in err:
        return False, r.stderr.strip(), "partition"
    if "invalid account" in err or ("account" in err and "not permitted" in err) or "invalid qos" in err:
        return False, r.stderr.strip(), "account"
    if "time limit" in err or "requested time" in err:
        return False, r.stderr.strip(), "time"
    return False, r.stderr.strip() or "sbatch rejected the job for an unknown reason.", ""


# --------------------------------------------------------------------------- #
#  Full validation of a loaded config
# --------------------------------------------------------------------------- #
def validate(cfg, live_probe=True):
    """Run every check we can. Returns True if there are no hard failures."""
    problems = 0

    # --- scheduler ---
    head("Scheduler")
    sched = (cfg.get("scheduler") or {})
    stype = sched.get("type", "")
    if stype not in ("slurm", "pbs", "sge", "local"):
        bad(f"scheduler.type is '{stype}' -- must be slurm, pbs, sge, or local.")
        problems += 1
    else:
        ok(f"scheduler.type = {stype}")

    if stype == "slurm":
        for tool in ("sbatch", "squeue", "scancel"):
            if shutil.which(tool):
                ok(f"{tool} found on PATH")
            else:
                bad(f"{tool} not found on PATH -- SLURM may not be available here.")
                problems += 1
        sl = sched.get("slurm") or {}
        gp = sl.get("gpu_partition", "")
        acct = sl.get("account", "")
        extra = sl.get("extra_flags", "")
        print(f"    gpu_partition={gp!r}  account={acct!r}  extra_flags={extra!r}")
        if live_probe and shutil.which("sbatch"):
            head("SLURM live probe (submits + cancels a 1-min dummy job)")
            good, msg, hint = probe_slurm(gp, acct, extra, want_gpu=True)
            if good:
                ok(msg)
            else:
                bad(msg)
                problems += 1
                if hint == "blank_gpu_partition":
                    warn("Your cluster forbids naming the GPU partition (e.g. Della). "
                         "Set gpu_partition to \"\" -- GPUs are requested via --gres.")
                elif hint == "account":
                    warn("Looks like an account/allocation is required or wrong. "
                         "Set slurm.account to your allocation.")
                elif hint == "partition":
                    warn("The GPU partition name looks wrong. Run `sinfo -o \"%P %G\"` "
                         "to see which partition actually has GPUs.")
                elif hint == "time":
                    warn("Requested time exceeds the partition limit. Lower the `hours` "
                         "values under resources: in config.yaml.")

    # --- conda environments ---
    head("Conda environments")
    envs = cfg.get("environments") or {}
    have = _conda_envs()
    if have is None:
        warn("could not run `conda env list` -- skipping env checks.")
    else:
        for key in ("bindcraft", "pesto", "rfd3"):
            name = envs.get(key)
            if not name:
                warn(f"environments.{key} is not set.")
            elif name in have:
                ok(f"environments.{key} = {name} (found)")
            else:
                warn(f"environments.{key} = {name} -- not in `conda env list`.")

    # --- paths ---
    head("Paths")
    paths = cfg.get("paths") or {}

    def check_path(key, required):
        val = paths.get(key, "")
        if not val:
            (bad if required else warn)(f"paths.{key} is empty.")
            return required
        if "/path/to/" in val:
            warn(f"paths.{key} still has a placeholder value: {val}")
            return False
        if Path(val).exists():
            ok(f"paths.{key} -> {val}")
            return False
        (bad if required else warn)(f"paths.{key} does not exist: {val}")
        return required

    problems += bool(check_path("bindcraft_dir", required=True))
    if paths.get("pesto_dir"):
        check_path("pesto_dir", required=False)
    # weights only matter for the RFdiffusion half -- warn, don't fail
    for key in ("rfd3_weights", "rfd3_checkpoint", "foundry_weights_dir", "rf3_weights", "ccd_mirror"):
        if paths.get(key):
            check_path(key, required=False)

    # workspace must be writable
    ws = paths.get("workspace", "")
    if ws:
        try:
            Path(ws).mkdir(parents=True, exist_ok=True)
            probe = Path(ws) / ".symplify_write_test"
            probe.write_text("ok")
            probe.unlink()
            ok(f"workspace is writable: {ws}")
        except Exception as e:  # noqa: BLE001
            bad(f"workspace not writable ({ws}): {e}")
            problems += 1
    else:
        bad("paths.workspace is empty.")
        problems += 1

    # --- summary ---
    head("Summary")
    if problems == 0:
        ok("No blocking problems found. You're good to launch.")
        return True
    bad(f"{problems} blocking problem(s) found -- fix the [fail] items above before launching.")
    return False


def _conda_envs():
    if not shutil.which("conda"):
        return None
    try:
        r = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    names = set()
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split()[0])
        # also capture the basename of full-path envs
        names.add(Path(line.split()[0]).name)
    return names


# --------------------------------------------------------------------------- #
#  Interactive setup
# --------------------------------------------------------------------------- #
def run_setup():
    print(_c("36;1", "\n=== Symplify setup ==="))
    cfg = load_config() or {}
    sched = cfg.get("scheduler") or {}
    sl = sched.get("slurm") or {}

    head("Which job scheduler does this cluster use?")
    stype = ask("slurm / pbs / sge / local", sched.get("type", "slurm")).lower()

    if stype != "slurm":
        warn("This wizard currently only auto-configures SLURM. "
             "Set the other fields in config.yaml by hand, then run --check.")
        return

    head("SLURM settings")
    print("  Tip: leave the GPU partition BLANK if your cluster routes GPU jobs")
    print("  automatically via --gres and forbids naming the partition (e.g. Della).")
    gpu_partition = ask("GPU partition name (blank = gres-only)", sl.get("gpu_partition", ""))
    cpu_partition = ask("CPU partition name", sl.get("cpu_partition", "cpu"))
    account = ask("Account / allocation (blank if not required)", sl.get("account", ""))
    extra_flags = ask("Extra SBATCH flags, e.g. --qos=high (blank for none)", sl.get("extra_flags", ""))

    # ---- validate live, auto-correct the Della case ----
    if shutil.which("sbatch"):
        head("Testing these settings against the scheduler...")
        good, msg, hint = probe_slurm(gpu_partition, account, extra_flags, want_gpu=True)
        if good:
            ok(msg)
        else:
            bad(msg)
            if hint == "blank_gpu_partition" and gpu_partition:
                warn("This cluster forbids naming the GPU partition. Retrying with it blank...")
                gpu_partition = ""
                good, msg, _ = probe_slurm(gpu_partition, account, extra_flags, want_gpu=True)
                if good:
                    ok("Success with GPU partition blank -- saving it that way.")
                else:
                    bad(msg)
                    if not ask_yes("Settings still failing. Save anyway?", default=False):
                        print("  Aborted -- nothing written.")
                        return
            elif not ask_yes("Settings failed validation. Save anyway?", default=False):
                print("  Aborted -- nothing written.")
                return
    else:
        warn("sbatch not on PATH here -- skipping the live test. "
             "Run `python setup_wizard.py --check` on a node with SLURM to verify.")

    done = write_slurm_settings({
        "gpu_partition": gpu_partition,
        "cpu_partition": cpu_partition,
        "account": account,
        "extra_flags": extra_flags,
    })
    ok(f"Wrote {', '.join(sorted(done))} to config.yaml")

    # ---- full validation pass on the freshly written config ----
    head("Running full validation...")
    validate(load_config(), live_probe=False)  # already probed above


def check_only():
    cfg = load_config()
    if cfg is None:
        return False
    return validate(cfg, live_probe=True)


# --------------------------------------------------------------------------- #
def main():
    if "--check" in sys.argv:
        sys.exit(0 if check_only() else 1)
    run_setup()


if __name__ == "__main__":
    main()