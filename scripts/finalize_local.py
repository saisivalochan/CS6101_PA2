#!/usr/bin/env python3
"""Finish local measurements with an isolated NQ timing validation.

The first NQ build overlapped a 55.009-second integration-test run. Preserve that
evidence, rebuild with identical parameters, verify the same graph, and promote
the clean measurement before repeating the five NQ query points. No paid services
are used. Completed isolated builds/checkpoints are reusable after interruption.
"""
import argparse
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone

from pa2.util import ROOT, configure_threads, index_id, load_config, read_json, sha256, write_json

# VAMANA02 writes these fields individually, without struct padding. Only the
# last field (elapsed construction seconds) varies between deterministic builds.
HEADER = struct.Struct("<8sIQIIIIIdQIQd")


def normalized_index_sha256(path):
    import hashlib
    with pathlib.Path(path).open("rb") as stream:
        raw = stream.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("Truncated Vamana header")
        values = HEADER.unpack(raw)
        if values[0] != b"VAMANA02" or values[1] != 0x01020304:
            raise ValueError("Unsupported Vamana format")
        digest = hashlib.sha256(HEADER.pack(*values[:-1], 0.0))
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def archive_file(source, destination):
    """Keep an exact original, refusing to overwrite conflicting prior evidence."""
    source, destination = pathlib.Path(source), pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.exists() and sha256(source) != sha256(destination):
            raise ValueError(f"Conflicting archived evidence: {destination}")
    elif source.exists():
        shutil.copyfile(source, destination)
        if sha256(source) != sha256(destination):
            raise ValueError(f"Archive copy checksum mismatch: {destination}")
    else:
        raise FileNotFoundError(source)
    return destination


def main(args):
    configure_threads()
    config = load_config(args.config)
    results = pathlib.Path(config["results_dir"])
    validation = results / "build_validation"
    validation.mkdir(parents=True, exist_ok=True)
    status_path = validation / "status.json"
    deadline = time.monotonic() + args.max_hours * 3600
    params = {k: config["vamana"][k] for k in ("R", "L_build", "alpha")}
    name = index_id("vamana", params)
    key = "nq__" + name
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(ROOT / ".tools/matplotlib")
    env["HF_HUB_CACHE"] = str(ROOT / "data/hf-hub-cache")

    def status(phase, **extra):
        value = {"phase": phase, "recorded_utc": datetime.now(timezone.utc).isoformat(),
                 "paid_operations_started": False, **extra}
        write_json(status_path, value)
        print(json.dumps(value), flush=True)

    def command(label, argv, progress_path=None):
        log = validation / "logs" / (label + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        status(label, command=argv, log=str(log.relative_to(ROOT)))
        with log.open("a") as output:
            output.write(json.dumps({"command": argv, "started_epoch": time.time()}) + "\n")
            output.flush()
            process = subprocess.Popen(argv, env=env, stdout=output, stderr=subprocess.STDOUT)
            try:
                last = 0.0
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        process.terminate()
                        process.wait(timeout=30)
                        raise RuntimeError("Finalization runtime exhausted; saved checkpoints are preserved")
                    if time.monotonic() - last >= 50:
                        progress = read_json(progress_path) if progress_path and progress_path.exists() else {}
                        status(label, log=str(log.relative_to(ROOT)), progress=progress)
                        last = time.monotonic()
                    time.sleep(5)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
            output.write(json.dumps({"finished_epoch": time.time(), "exit_code": process.returncode}) + "\n")
        if process.returncode:
            raise RuntimeError(f"{label} failed; inspect {log}")

    clean_marker = validation / "clean_nq_validation.json"
    if not clean_marker.exists():
        status("waiting_for_original_pipeline")
        last = 0.0
        while True:
            audit_path = results / "audit.json"
            if audit_path.exists() and read_json(audit_path).get("complete"):
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Original pipeline has not completed within the runtime allowance")
            if time.monotonic() - last >= 50:
                path = results / "progress" / (key + ".json")
                status("waiting_for_original_pipeline", progress=read_json(path) if path.exists() else {})
                last = time.monotonic()
            time.sleep(5)

        # Verify the actual full experimental evidence before changing artifacts.
        from pa2.audit import audit
        summary, _ = audit(config, verify_checksums=True)
        if not summary["complete"]:
            raise RuntimeError("Original experiment audit failed")
        canonical_index = pathlib.Path(config["index_dir"]) / "nq" / (name + ".vamana")
        canonical_meta = results / "builds" / (key + ".json")
        original = validation / "original"
        archive_file(canonical_meta, original / "nq_build.json")
        archived_index = pathlib.Path(config["index_dir"]) / "build_validation/original/nq.vamana"
        archive_file(canonical_index, archived_index)
        if sha256(archived_index) != read_json(original / "nq_build.json")["index_sha256"]:
            raise ValueError("Original NQ index does not match its saved metadata")

        isolated = {k: v for k, v in config.items() if k != "config_path"}
        isolated["index_dir"] = str(pathlib.Path(config["index_dir"]) / "build_validation/isolated")
        isolated["results_dir"] = str(validation / "isolated")
        isolated_config = validation / "isolated_config.json"
        write_json(isolated_config, isolated)
        base = [sys.executable, "-m", "pa2.cli", "worker", "--config", str(isolated_config)]
        command("isolated_nq_build", base + ["--task", "build", "--dataset", "nq", "--method", "vamana",
                "--params", json.dumps(params)],
                pathlib.Path(isolated["results_dir"]) / "progress" / (key + ".json"))
        clean_index = pathlib.Path(isolated["index_dir"]) / "nq" / (name + ".vamana")
        clean_meta = read_json(pathlib.Path(isolated["results_dir"]) / "builds" / (key + ".json"))
        original_meta = read_json(original / "nq_build.json")
        graph_hash = normalized_index_sha256(archived_index)
        if graph_hash != normalized_index_sha256(clean_index):
            raise ValueError("Isolated rebuild differs beyond the elapsed-time header; originals preserved")
        if clean_meta["signature"] != original_meta["signature"] or sha256(clean_index) != clean_meta["index_sha256"]:
            raise ValueError("Isolated NQ provenance/checksum mismatch")

        for setting in config["vamana"]["L_search"]:
            filename = key + "__" + str(setting) + ".json"
            for folder in ("points", "timing"):
                path = results / folder / filename
                archived = archive_file(path, original / folder / filename)
                if sha256(path) != sha256(archived):
                    raise ValueError("Original operating-point copy was not verified")
                path.unlink()  # Verified exact copies remain under original/.
        temporary = canonical_index.with_suffix(".replacement")
        shutil.copyfile(clean_index, temporary)
        if sha256(temporary) != clean_meta["index_sha256"]:
            raise ValueError("Promoted NQ index copy checksum mismatch")
        temporary.replace(canonical_index)
        note = ("The initial build overlapped a 55.009-second integration-test run on 2026-10-16. "
                "The reported construction time is from a subsequent isolated rebuild with identical "
                "data, parameters and seed. The two serialized graphs match byte-for-byte after "
                "normalizing only the elapsed-time header. Initial measurements are retained under "
                "results/build_validation/original; NQ query timings were repeated after the clean build.")
        clean_meta["measurement_notes"] = [note]
        clean_meta["validation_history"] = {
            "initial_build_seconds": original_meta["build_seconds"],
            "isolated_build_seconds": clean_meta["build_seconds"],
            "initial_index_sha256": original_meta["index_sha256"],
            "graph_sha256_without_elapsed": graph_hash}
        write_json(canonical_meta, clean_meta)
        write_json(clean_marker, clean_meta["validation_history"])

    # All native construction has finished before running any query timing here.
    base = [sys.executable, "-m", "pa2.cli", "worker", "--config", config["config_path"]]
    for setting in config["vamana"]["L_search"]:
        command("evaluate_nq_" + str(setting), base + ["--task", "evaluate", "--dataset", "nq", "--method", "vamana",
                "--params", json.dumps(params), "--setting", str(setting)])
    command("final_tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    from pa2.audit import audit, submission_details
    summary, _ = audit(config, verify_checksums=True)
    if not summary["complete"]:
        raise RuntimeError("Final experiment audit failed: " + "; ".join(summary["missing_or_invalid"]))
    command("export_llm_chats", [sys.executable, "scripts/export_llm_chats.py"])
    from pa2.report import generate, package
    details = pathlib.Path(args.details)
    _, missing = submission_details(details)
    report_details = details if not missing else ROOT / "submission_details.draft.json"
    report = generate(config, report_details, draft=bool(missing))
    archive = package(config, details) if not missing else None
    audit(config, verify_checksums=True)  # Keep experiment-only audit distinct from disclosure status.
    final = {"experiments_complete": True, "available_operating_points": summary["available_operating_points"],
             "report": str(report.relative_to(ROOT)), "submission_complete": not missing,
             "submission_archive": str(archive.relative_to(ROOT)) if archive else None,
             "missing_submission_details": missing, "tests_exit_code": 0, "paid_operations_started": False,
             "recorded_utc": datetime.now(timezone.utc).isoformat()}
    write_json(results / "local_execution_status.json", final)
    write_json(results / "submission_status.json", final)
    with (ROOT / "docs/progress.md").open("a") as stream:
        stream.write("\n## Local experimental completion — " + final["recorded_utc"] + "\n\n")
        stream.write("The full experiment audit passes with 48 required operating points. All full corpora were used. "
                     "A clean isolated NQ rebuild replaced the initial overlapping construction measurement, "
                     "with graph equivalence verified and initial evidence preserved. All tests passed after "
                     "construction and query measurement finished.\n\n")
        stream.write("Report: `" + final["report"] + "`. Submission complete: " + str(final["submission_complete"]) + ".\n")
        if missing:
            stream.write("Remaining submission requirements: " + "; ".join(missing) + ".\n")
    status("finished", **final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.json")
    parser.add_argument("--details", default="submission_details.json")
    parser.add_argument("--max-hours", type=float, default=12)
    args = parser.parse_args()
    if args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    try:
        main(args)
    except BaseException as error:
        print(f"Finalization stopped: {error}", file=sys.stderr, flush=True)
        raise
