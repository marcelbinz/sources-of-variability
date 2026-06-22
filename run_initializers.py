#!/usr/bin/env python3
"""
Run all python scripts starting with
  - initialize-train-itc-
  - initialize-train-mm-

Usage:
  python run_initializers.py [--dir PATH] [--stop-on-error]

By default the search is recursive from `--dir` (default: current directory).
The scripts are executed using the same Python interpreter running this script.
"""

from pathlib import Path
import subprocess
import sys
import argparse


def find_scripts(root: Path):
    patterns = ("initialize-train-itc-*.py", "initialize-train-mm-*.py")
    scripts = []
    for pat in patterns:
        scripts.extend(sorted(root.rglob(pat)))
    # remove duplicates and sort by path
    scripts = sorted(dict.fromkeys(scripts))
    return scripts


def run_script(path: Path, python_exe: str):
    print(f"\n=== Running: {path} ===")
    # run with cwd set to script's parent so relative paths inside the script work
    proc = subprocess.run([python_exe, str(path)], cwd=path.parent)
    return proc.returncode


def main():
    p = argparse.ArgumentParser(description="Run initializer scripts for ITC and MM")
    p.add_argument("--dir", "-d", default=".", help="Root directory to search (default: current dir)")
    p.add_argument("--stop-on-error", action="store_true", help="Stop when a script returns non-zero exit code")
    p.add_argument("--list-only", action="store_true", help="Only list found scripts without executing them")
    # conda environment hard-coded below; CLI option removed
    args = p.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists():
        print(f"Directory not found: {root}")
        sys.exit(2)

    scripts = find_scripts(root)
    if not scripts:
        print("No matching initializer scripts found.")
        return

    print(f"Found {len(scripts)} scripts:")
    for s in scripts:
        print(" -", s)

    if args.list_only:
        return

    # Hard-coded conda environment name
    conda_env = "SoV"

    results = []
    for s in scripts:
        if conda_env:
            cmd = ["conda", "run", "-n", conda_env, "python", str(s)]
            print(f"\n=== Running in conda env '{conda_env}': {s} ===")
            proc = subprocess.run(cmd, cwd=s.parent)
            rc = proc.returncode
        else:
            python_exe = sys.executable
            rc = run_script(s, python_exe)
        results.append((s, rc))
        if rc != 0:
            print(f"Script {s} exited with code {rc}")
            if args.stop_on_error:
                break

    ok = [s for s, rc in results if rc == 0]
    bad = [s for s, rc in results if rc != 0]

    print("\nSummary:")
    print(f"  succeeded: {len(ok)}")
    print(f"  failed:    {len(bad)}")
    if bad:
        for s, rc in results:
            if rc != 0:
                print(f"  - {s} (exit {rc})")


if __name__ == "__main__":
    main()
