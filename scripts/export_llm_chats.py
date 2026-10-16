#!/usr/bin/env python3
"""Export visible project chat records for the assignment's LLM disclosure.

Exports user/assistant messages and tool calls/results from matching local Codex
sessions. Internal reasoning and system/developer messages are excluded. Active
sessions are snapshots: rerun this after the last assignment conversation ends.
This cannot discover chats in other products or create public shareable URLs.
"""
import argparse
import base64
import hashlib
import json
import os
import pathlib
import re
from datetime import datetime, timezone


def preserve_image_attachments(text, output_dir):
    """Keep visible images as files instead of printing their binary encoding."""
    def replace(match):
        media, encoded = match.groups()
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError:
            return match.group(0)
        digest = hashlib.sha256(raw).hexdigest()
        extension = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}[media]
        relative = pathlib.Path("attachments") / (digest + "." + extension)
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != raw:
            raise ValueError("Conflicting image attachment")
        target.write_bytes(raw)
        return f"[Binary attachment: {relative.as_posix()}; media={media}; sha256={digest}]"
    return re.sub(r"data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)", replace, text)


def export(session_root, workspace, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = str(workspace.resolve()).rstrip("/").casefold()
    entries = []
    for source in sorted(session_root.rglob("*.jsonl")):
        with source.open() as stream:
            first = stream.readline()
        try:
            metadata = json.loads(first)
        except ValueError:
            continue
        if metadata.get("type") != "session_meta":
            continue
        meta = metadata.get("payload", {})
        if str(meta.get("cwd", "")).rstrip("/").casefold() != expected:
            continue
        snapshot = source.read_bytes()
        identifier = str(meta.get("id", source.stem))
        lines = ["Codex project conversation", f"Session: {identifier}",
                 "Export: visible messages and tool activity; internal reasoning and",
                 "system/developer instructions are excluded.",
                 "Inline image encodings are preserved as referenced image attachments.", ""]
        count = 0
        for record in snapshot.decode("utf-8").splitlines():
            try:
                value = json.loads(record)
            except ValueError:
                continue  # An active session can end with an unfinished record.
            if value.get("type") != "response_item":
                continue
            item = value.get("payload", {})
            if item.get("channel") == "analysis":
                continue
            kind = item.get("type")
            if kind == "message" and item.get("role") in ("user", "assistant"):
                text = "\n".join(part.get("text", "[Non-text attachment]")
                                 for part in item.get("content", []))
                label = item["role"].upper()
            elif kind in ("function_call", "custom_tool_call"):
                label = "TOOL CALL: " + item.get("name", "unknown")
                text = item.get("arguments", item.get("input", ""))
            elif kind in ("function_call_output", "custom_tool_call_output"):
                label = "TOOL RESULT: " + item.get("call_id", "")
                text = item.get("output", "")
            else:
                continue
            if not isinstance(text, str):
                text = json.dumps(text, ensure_ascii=False, indent=2)
            text = preserve_image_attachments(text, output_dir)
            lines.extend([f"[{value.get('timestamp', '')}] {label}", text, ""])
            count += 1
        target = output_dir / f"codex_{identifier}.txt"
        content = "\n".join(lines) + "\n"
        target.write_text(content)
        entries.append({"session_id": identifier, "transcript_path": target.name,
                        "records": count, "source_snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
                        "transcript_sha256": hashlib.sha256(content.encode()).hexdigest()})
    attachments = [{"path": p.relative_to(output_dir).as_posix(),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in sorted((output_dir / "attachments").glob("*"))] if (output_dir / "attachments").exists() else []
    manifest = {"exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "complete_disclosure_confirmed": False,
                "limitations": ["Active conversations require a fresh export after completion.",
                                "Other products and sessions in other directories require separate disclosure.",
                                "Genuine shareable chat/transcript URLs must be supplied by the user."],
                "sessions": entries, "image_attachments": attachments}
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=pathlib.Path,
                        default=pathlib.Path(os.environ.get("CODEX_HOME", pathlib.Path.home() / ".codex")) / "sessions")
    parser.add_argument("--workspace", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("llm_transcripts"))
    args = parser.parse_args()
    print(json.dumps(export(args.session_root, args.workspace, args.output_dir), indent=2))
