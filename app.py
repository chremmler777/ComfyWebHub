#!/usr/bin/env python3
"""Keeper web - mark favorites, tag HQ pass type, export notes for Claude."""
import json
import os
import re
from pathlib import Path
from flask import Flask, jsonify, request, send_file, send_from_directory

OUTPUT_ROOT = Path("/home/chremmler/ComfyUI/output/comfy")
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static")


def data_file(character: str) -> Path:
    return DATA_DIR / f"{character}.json"


def load_marks(character: str) -> dict:
    f = data_file(character)
    if f.exists():
        return json.loads(f.read_text())
    return {}


def save_marks(character: str, marks: dict) -> None:
    data_file(character).write_text(json.dumps(marks, indent=2, sort_keys=True))


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/characters")
def characters():
    if not OUTPUT_ROOT.exists():
        return jsonify([])
    chars = []
    for p in sorted(OUTPUT_ROOT.iterdir()):
        if p.is_dir():
            png_count = sum(1 for _ in p.glob("*.png"))
            if png_count > 0:
                chars.append({"name": p.name, "count": png_count})
    return jsonify(chars)


REFINE_RE = re.compile(r"_(refine|hq)_(\d+)_")


@app.get("/api/images/<character>")
def images(character: str):
    d = OUTPUT_ROOT / character
    if not d.is_dir():
        return jsonify([]), 404
    sort = request.args.get("sort", "newest")
    files = list(d.glob("*.png"))
    # Map source number -> list of refine/hq file stems (e.g. "113" -> [aria_refine_113_00001, ...])
    refine_by_num: dict[str, list[str]] = {}
    for f in files:
        m = REFINE_RE.search(f.stem)
        if m:
            num = m.group(2).lstrip("0") or "0"
            refine_by_num.setdefault(num, []).append(f.stem)
    # Map refine stem -> original stem (look up original by number)
    orig_by_num: dict[str, str] = {}
    plain_re = re.compile(rf"^{re.escape(character)}_0*(\d+)_$")
    for f in files:
        m = plain_re.match(f.stem)
        if m:
            orig_by_num[m.group(1).lstrip("0") or "0"] = f.stem
    if sort == "newest":
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        files.sort(key=lambda p: p.name)
    marks = load_marks(character)
    out = []
    for f in files:
        st = f.stat()
        mark = marks.get(f.stem, {})
        stem = f.stem
        is_refined = False
        pair = None
        # Is this a refine/hq output? -> pair is the original
        rm = REFINE_RE.search(stem)
        if rm:
            num = rm.group(2).lstrip("0") or "0"
            pair = orig_by_num.get(num)
        else:
            # Plain image — check if it has refine siblings
            pm = plain_re.match(stem)
            if pm:
                num = pm.group(1).lstrip("0") or "0"
                siblings = refine_by_num.get(num, [])
                if siblings:
                    is_refined = True
                    pair = sorted(siblings)[0]
        out.append({
            "name": stem,
            "file": f.name,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "keep": mark.get("keep", False),
            "reject": mark.get("reject", False),
            "hq": mark.get("hq"),
            "note": mark.get("note", ""),
            "refined": is_refined,
            "pair": pair,
        })
    return jsonify(out)


@app.get("/img/<character>/<path:filename>")
def serve_img(character: str, filename: str):
    path = OUTPUT_ROOT / character / filename
    if not path.is_file():
        return "not found", 404
    return send_file(path)


@app.post("/api/mark/<character>/<name>")
def mark(character: str, name: str):
    body = request.get_json(force=True) or {}
    marks = load_marks(character)
    entry = marks.get(name, {})
    for k in ("keep", "reject", "hq", "note"):
        if k in body:
            entry[k] = body[k]
    # clean up empty
    entry = {k: v for k, v in entry.items() if v not in (False, None, "")}
    if entry:
        marks[name] = entry
    else:
        marks.pop(name, None)
    save_marks(character, marks)
    return jsonify({"ok": True, "entry": entry})


def build_export(character: str) -> str:
    marks = load_marks(character)
    keepers = sorted(k for k, v in marks.items() if v.get("keep"))
    hq_groups = {}
    for k, v in marks.items():
        if v.get("keep") and v.get("hq"):
            hq_groups.setdefault(v["hq"], []).append(k)
    noted = sorted(k for k, v in marks.items() if v.get("note") and not v.get("keep") and not v.get("reject"))
    lines = [f"# {character} — {len(keepers)} keepers, {len(noted)} refine", ""]
    if keepers:
        lines.append("## Keepers")
        for k in keepers:
            note = marks[k].get("note", "")
            hq = marks[k].get("hq", "")
            bits = []
            if hq:
                bits.append(f"hq={hq}")
            if note:
                bits.append(note)
            tail = f" — {', '.join(bits)}" if bits else ""
            lines.append(f"- {k}{tail}")
        lines.append("")
    if noted:
        lines.append("## Refine (noted but not kept)")
        for k in noted:
            lines.append(f"- {k} — {marks[k]['note']}")
        lines.append("")
    if hq_groups:
        lines.append("## HQ Pass Queue")
        for hq_type, names in sorted(hq_groups.items()):
            lines.append(f"### {hq_type}")
            for n in sorted(names):
                lines.append(f"- {n}")
    return "\n".join(lines)


@app.get("/api/export/<character>")
def export(character: str):
    return build_export(character), 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.post("/api/submit/<character>")
def submit(character: str):
    body = build_export(character)
    path = DATA_DIR / f"{character}_submit.md"
    path.write_text(body)
    marks = load_marks(character)
    keepers = sum(1 for v in marks.values() if v.get("keep"))
    # delete rejected images from disk, clear their entries
    char_dir = OUTPUT_ROOT / character
    deleted = 0
    remaining = {}
    for name, entry in marks.items():
        if entry.get("reject"):
            png = char_dir / f"{name}.png"
            if png.is_file():
                png.unlink()
                deleted += 1
        else:
            remaining[name] = entry
    save_marks(character, remaining)
    # If character folder is now empty, remove it
    folder_removed = False
    if char_dir.is_dir() and not any(char_dir.iterdir()):
        char_dir.rmdir()
        folder_removed = True
    return jsonify({"ok": True, "path": str(path), "keepers": keepers, "deleted": deleted, "folder_removed": folder_removed})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5151, debug=False)
