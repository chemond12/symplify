"""
run.py
------
Symplify server entry point.

Usage:
  python run.py           — start the server
  python run.py --check   — validate configuration and exit
  python run.py --port N  — override port from config
"""

import argparse
import os
import sys
from pathlib import Path

import yaml


SYMPLIFY_DIR = Path(__file__).resolve().parent


def load_config():
    cfg_path = SYMPLIFY_DIR / "config.yaml"
    if not cfg_path.exists():
        print(f"[ERROR] config.yaml not found at {cfg_path}")
        print("  Copy config.yaml.example to config.yaml and edit it.")
        sys.exit(1)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def check_config(cfg):
    """Validate configuration and print a report."""
    errors   = []
    warnings = []
    ok       = []

    paths = cfg.get("paths", {})
    envs  = cfg.get("environments", {})

    # Check workspace
    ws = paths.get("workspace", "")
    if not ws:
        errors.append("paths.workspace is not set")
    else:
        p = Path(ws)
        try:
            p.mkdir(parents=True, exist_ok=True)
            ok.append(f"workspace: {ws}")
        except Exception as e:
            errors.append(f"Cannot create workspace at {ws}: {e}")

    # Check tool paths
    bindcraft = paths.get("bindcraft_dir", "")
    if bindcraft and Path(bindcraft).exists():
        ok.append(f"BindCraft: {bindcraft}")
    else:
        warnings.append(f"bindcraft_dir not found: {bindcraft or '(not set)'}")

    pesto = paths.get("pesto_dir", "")
    if pesto and Path(pesto).exists():
        ok.append(f"PESTO: {pesto}")
    else:
        warnings.append("PESTO not configured — will use fallback hotspot detection")

    ccd = paths.get("ccd_mirror", "")
    if ccd and Path(ccd).exists():
        ok.append(f"CCD mirror: {ccd}")
    else:
        warnings.append(f"CCD mirror not found: {ccd or '(not set)'}")

    # Check scheduler type
    sched = cfg.get("scheduler", {}).get("type", "slurm")
    ok.append(f"Scheduler: {sched}")

    # Check frontend
    frontend = SYMPLIFY_DIR / "frontend" / "index.html"
    if frontend.exists():
        ok.append("Frontend: found")
    else:
        errors.append("Frontend not found — run from the symplify directory")

    # Check Python packages
    for pkg in ["flask", "yaml"]:
        try:
            __import__(pkg if pkg != "yaml" else "yaml")
            ok.append(f"Python: {pkg} ✓")
        except ImportError:
            errors.append(f"Missing Python package: {pkg} (pip install {pkg})")

    # Report
    print("\n" + "="*50)
    print(" Symplify Configuration Check")
    print("="*50)
    for msg in ok:      print(f"  ✓  {msg}")
    for msg in warnings: print(f"  ⚠  {msg}")
    for msg in errors:   print(f"  ✗  {msg}")
    print("="*50)

    if errors:
        print(f"\n{len(errors)} error(s) — fix before running Symplify.\n")
        return False
    else:
        print(f"\nConfiguration OK. Run `python run.py` to start.\n")
        return True


def main():
    parser = argparse.ArgumentParser(description="Symplify — Binder Design Platform")
    parser.add_argument("--check", action="store_true",
                        help="Validate configuration and exit")
    parser.add_argument("--port", type=int, default=None,
                        help="Override port from config.yaml")
    parser.add_argument("--host", default=None,
                        help="Override host from config.yaml")
    args = parser.parse_args()

    cfg = load_config()

    if args.check:
        ok = check_config(cfg)
        sys.exit(0 if ok else 1)

    # Add backend to path
    sys.path.insert(0, str(SYMPLIFY_DIR / "backend"))
    sys.path.insert(0, str(SYMPLIFY_DIR / "pipeline" / "core"))

    # Copy frontend to static dir Flask expects
    import shutil
    src = SYMPLIFY_DIR / "frontend" / "index.html"
    dst = SYMPLIFY_DIR / "frontend" / "dist"
    dst.mkdir(exist_ok=True)
    shutil.copy2(src, dst / "index.html")

    from app import app

    server = cfg.get("server", {})
    host   = args.host or server.get("host", "127.0.0.1")
    port   = args.port or server.get("port", 8080)
    debug  = server.get("debug", False)

    print(f"""
╔══════════════════════════════════════════╗
║          Symplify — iGEM 2026            ║
╠══════════════════════════════════════════╣
║  Server:    http://{host}:{port}
║  Scheduler: {cfg.get('scheduler',{}).get('type','slurm').upper()}
║
║  Access via SSH tunnel:
║    ssh -L {port}:localhost:{port} user@cluster
║  Then open:
║    http://localhost:{port}
╚══════════════════════════════════════════╝
""")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
