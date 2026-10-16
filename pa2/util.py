"""Artifact provenance and runtime measurements shared by experiments."""
import csv
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import resource
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())


def write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    config = read_json(path)
    config["config_path"] = str(pathlib.Path(path).resolve())
    for key in ("data_dir", "index_dir", "results_dir"):
        config[key] = str(pathlib.Path(config[key]).resolve())
    if config["threads"] != 1:
        raise ValueError("This benchmark policy requires one CPU thread")
    if config["pca_components"] != 128:
        raise ValueError("The compulsory assignment requires 128 search dimensions")
    if config["timing_queries"] < 1000 or config["timing_repeats"] < 3:
        raise ValueError("Use at least 1000 timing queries when available and three repetitions")
    if len(set(config["hnsw"]["efSearch"])) < 5 or len(set(config["vamana"]["L_search"])) < 5:
        raise ValueError("At least five distinct search settings are required")
    if len(config["study"]) < 3:
        raise ValueError("At least three development graphs are required")
    return config


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def machine_info():
    cpu = platform.processor()
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    versions = {}
    for package in ("numpy", "faiss-cpu", "scikit-learn", "datasets", "matplotlib", "reportlab", "threadpoolctl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    result = {"cpu":cpu,"host":platform.node(),"logical_cpus":os.cpu_count(),"platform":platform.platform(),
              "python":sys.version,"packages":versions,"recorded_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    compiler = ROOT / "build/compiler.json"
    if compiler.exists():
        result["core_compiler"] = read_json(compiler)
    return result


def configure_threads():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


def save_csv(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def index_id(method, params=None):
    params=params or {}
    if method=="vamana":
        return f"vamana_R{params['R']}_B{params['L_build']}_a{params['alpha']:g}"
    if method=="hnsw":
        return f"hnsw_M{params['M']}_C{params['efConstruction']}"
    if method=="exact":
        return "exact"
    raise ValueError(f"unknown method: {method}")
