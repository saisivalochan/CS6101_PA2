#!/usr/bin/env python3
"""Build the C++ core without Python headers or third-party packages."""
import argparse
import json
import pathlib
import platform
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

def build(debug=False, sanitize=False):
    out = ROOT / "build"
    out.mkdir(exist_ok=True)
    flags = ["-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-fPIC", "-shared"]
    flags += ["-O0", "-g", "-D_GLIBCXX_ASSERTIONS"] if debug else ["-O3", "-march=native", "-DNDEBUG"]
    if sanitize:
        flags += ["-fsanitize=undefined", "-fno-sanitize-recover=undefined"]
    cmd = ["g++", *flags, str(ROOT / "cpp/vamana.cpp"), "-o", str(out / "libvamana.so")]
    subprocess.run(cmd, check=True)
    compiler = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
    (out / "compiler.json").write_text(json.dumps({"compiler": compiler, "command": cmd, "machine": platform.machine()}, indent=2))
    return out / "libvamana.so"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sanitize", action="store_true", help="Enable undefined-behavior sanitizer")
    args=parser.parse_args()
    print(build(args.debug,args.sanitize))
