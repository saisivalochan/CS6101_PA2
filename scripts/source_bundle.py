#!/usr/bin/env python3
"""Create a cloud-upload source bundle, explicitly separate from final submission."""
import hashlib
import pathlib
import tarfile

ROOT=pathlib.Path(__file__).resolve().parents[1]

def bundle():
    output=ROOT/"pa2-source.tar.gz"
    entries=[]
    for name in ("pa2","cpp","scripts","configs","cloud","tests","docs"):
        entries.extend(path for path in (ROOT/name).rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.name!="runtime.json" and path.suffix!=".pyc")
    entries += [ROOT/name for name in ("README.md","AGENTS.md","pyproject.toml","submission_details.example.json","ProblemStatement.pdf")]
    with tarfile.open(output,"w:gz") as archive:
        for path in sorted(entries):archive.add(path,arcname="CS6101_PA2/"+path.relative_to(ROOT).as_posix(),recursive=False)
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+".sha256").write_text(digest+"  "+output.name+"\n")
    print(f"{output}\nSHA256 {digest}\nSource upload only; this is not the assignment submission.")
    return output

if __name__=="__main__":bundle()
