#!/usr/bin/env python3
"""Keeper web - mark favorites, tag HQ pass type, export notes for Claude."""
import base64
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from flask import Flask, jsonify, request, send_file, send_from_directory, Response
from PIL import Image
import pipeline_detect

OUTPUT_ROOT = Path("/home/chremmler/ComfyUI/output/comfy")
DATA_DIR = Path(__file__).parent / "data"
VIDEO_DIR = Path("/home/chremmler/ComfyUI/output/videos")
RUNPOD_COMFY = os.environ.get("RUNPOD_COMFY", "https://ff55ciault2yrs-8188.proxy.runpod.net")
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID", "")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
DATA_DIR.mkdir(exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "video_cache").mkdir(exist_ok=True)
(DATA_DIR / "video_thumbs").mkdir(exist_ok=True)
pipeline_detect.init_cache(DATA_DIR)

# Animate job log: list of dicts, newest first (persisted to disk)
_animate_jobs: list[dict] = []
_animate_jobs_lock = threading.Lock()
JOBS_FILE = DATA_DIR / "animate_jobs.json"

# Always-on WAN content LoRAs (injected at dispatch regardless of planner output).
# SmoothFutanaris fixes cock/genital instability — recommended I2V weight 0.5-0.8.
DEFAULT_WAN_LORAS: list[tuple[str, float]] = [("SmoothFutanaris", 0.7)]


def _ensure_default_wan_loras(content_loras: list) -> list:
    """Merge DEFAULT_WAN_LORAS into a job's content_loras without duplicating by name."""
    merged = [list(x) for x in (content_loras or [])]
    have = {x[0] for x in merged}
    for name, strength in DEFAULT_WAN_LORAS:
        if name not in have:
            merged.append([name, strength])
    return merged


def _save_jobs():
    with _animate_jobs_lock:
        jobs_copy = list(_animate_jobs)
    JOBS_FILE.write_text(json.dumps(jobs_copy, indent=2))


def _load_saved_jobs() -> list:
    if JOBS_FILE.exists():
        try:
            return json.loads(JOBS_FILE.read_text())
        except Exception:
            return []
    return []


def _parse_mode(mode_str, fast_mode_fallback=True) -> tuple[bool, int]:
    """Parse mode string ('fast','q10','q20','q30') → (fast_mode, quality_steps)."""
    if mode_str is None:
        return (bool(fast_mode_fallback), 20)
    if mode_str == "fast":
        return (True, 6)
    if str(mode_str).startswith("q"):
        try:
            return (False, int(str(mode_str)[1:]))
        except ValueError:
            pass
    return (bool(fast_mode_fallback), 20)

app = Flask(__name__, static_folder="static")


@app.before_request
def _require_basic_auth():
    auth = request.authorization
    if not auth or auth.username != "lala" or auth.password != "lala":
        return Response("login required", 401, {"WWW-Authenticate": 'Basic realm="keeper-web"'})


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
    resp = send_from_directory("static", "index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/latest")
def latest_page():
    resp = send_from_directory("static", "latest.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/swipe")
def swipe_page():
    resp = send_from_directory("static", "swipe.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/api/latest")
def latest():
    if not OUTPUT_ROOT.exists():
        return jsonify([])
    limit = int(request.args.get("limit", "60"))
    all_files = []
    for p in OUTPUT_ROOT.iterdir():
        if not p.is_dir():
            continue
        char = p.name
        marks = load_marks(char)
        hq_done = hq_rendered_set(char, p)
        for f in p.glob("*.png"):
            st = f.stat()
            mark = marks.get(f.stem, {})
            pl = pipeline_detect.get_pipeline(char, f.stem, f)
            all_files.append({
                "character": char,
                "name": f.stem,
                "file": f.name,
                "mtime": st.st_mtime,
                "size": st.st_size,
                "keep": mark.get("keep", False),
                "reject": mark.get("reject", False),
                "hq": mark.get("hq"),
                "note": mark.get("note", ""),
                "regen": mark.get("regen", False),
                "hq_done": f.stem in hq_done,
                "civitai": mark.get("civitai", False),
                "stars": mark.get("stars", 0),
                "pipeline": pl.get("short", "?"),
                "pipeline_name": pl.get("name", ""),
            })
    all_files.sort(key=lambda x: x["mtime"], reverse=True)
    pipeline_detect.save_cache()
    return jsonify(all_files[:limit])


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


def hq_rendered_set(character: str, char_dir: Path) -> set[str]:
    """Return set of source-stems whose HQ pass has already been rendered on disk.

    Convention: `<char>_<N>_` → `<char>_hq_<N>_00001_` (plus variants like
    `<char>_<tag>_<N>_` → `<char>_hq_<tag>_<N>_` or `<char>_hq_<N>_`).
    This is a heuristic — we just check for any HQ file whose numeric suffix
    matches the source's numeric suffix, per-character.
    """
    done: set[str] = set()
    if not char_dir.is_dir():
        return done
    plain_re = re.compile(rf"^{re.escape(character)}_0*(\d+)_?$")
    # Collect numeric IDs that have an HQ render on disk.
    hq_ids: set[str] = set()
    hq_tag_ids: set[tuple[str, str]] = set()  # (tag, id)
    hq_re_plain = re.compile(rf"^{re.escape(character)}_hq_0*(\d+)_\d+_?$")
    hq_re_tag = re.compile(rf"^{re.escape(character)}_hq_([a-zA-Z][a-zA-Z0-9_]*?)_0*(\d+)_\d+_?$")
    for f in char_dir.glob(f"{character}_hq_*.png"):
        m = hq_re_tag.match(f.stem)
        if m:
            hq_tag_ids.add((m.group(1).lower(), m.group(2)))
            continue
        m = hq_re_plain.match(f.stem)
        if m:
            hq_ids.add(m.group(1))
    # For every source file, check if its numeric id has a matching HQ.
    for f in char_dir.glob("*.png"):
        stem = f.stem
        stem_nt = stem.rstrip("_")
        m = plain_re.match(stem)
        if m and m.group(1) in hq_ids:
            done.add(stem)
            continue
        # Tag variants: e.g. jade_test_00053 → hq=jade_hq_53 or hq=jade_hq_test_53
        parts = stem_nt.split("_")
        if len(parts) >= 3 and parts[0] == character and parts[-1].isdigit():
            num = parts[-1].lstrip("0") or "0"
            if num in hq_ids:
                done.add(stem)
                continue
            tag = "_".join(parts[1:-1]).lower()
            if tag and (tag, num) in hq_tag_ids:
                done.add(stem)
                continue
    return done


@app.get("/api/hq_status/<character>")
def hq_status(character: str):
    d = OUTPUT_ROOT / character
    return jsonify(sorted(hq_rendered_set(character, d)))


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
    hq_done = hq_rendered_set(character, d)
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
        pl = pipeline_detect.get_pipeline(character, stem, f)
        out.append({
            "name": stem,
            "file": f.name,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "keep": mark.get("keep", False),
            "reject": mark.get("reject", False),
            "hq": mark.get("hq"),
            "note": mark.get("note", ""),
            "regen": mark.get("regen", False),
            "hq_done": stem in hq_done,
            "refined": is_refined,
            "pair": pair,
            "civitai": mark.get("civitai", False),
            "stars": mark.get("stars", 0),
            "pipeline": pl.get("short", "?"),
            "pipeline_name": pl.get("name", ""),
            "scene": pl.get("scene", ""),
        })
    pipeline_detect.save_cache()
    return jsonify(out)


@app.get("/img/<character>/<path:filename>")
def serve_img(character: str, filename: str):
    path = OUTPUT_ROOT / character / filename
    if not path.is_file():
        return "not found", 404
    return send_file(path)


THUMB_ROOT = OUTPUT_ROOT.parent / ".keeper_thumbs"
THUMB_MAX_W = 720
THUMB_QUALITY = 78
BIG_ROOT = OUTPUT_ROOT.parent / ".keeper_big"
BIG_MAX_W = 1568
BIG_QUALITY = 85


def _serve_scaled(character: str, filename: str, cache_root, max_w: int, quality: int):
    """Downscaled JPEG of /img/<character>/<filename>, disk-cached under cache_root.

    Bandwidth saver for phones: serves a max-<max_w>px-wide JPEG instead of the
    full-res PNG. Reuses the cache file as long as it's newer than the source."""
    src = OUTPUT_ROOT / character / filename
    if not src.is_file():
        return "not found", 404

    cache = cache_root / character / (filename + ".jpg")
    try:
        if cache.is_file() and cache.stat().st_mtime >= src.stat().st_mtime:
            resp = send_file(cache, mimetype="image/jpeg")
            resp.headers["Cache-Control"] = "max-age=86400"
            return resp
    except OSError:
        pass

    try:
        img = Image.open(src)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (0, 0, 0))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
        if img.width > max_w:
            h = int(img.height * max_w / img.width)
            img = img.resize((max_w, h), Image.LANCZOS)
        cache.parent.mkdir(parents=True, exist_ok=True)
        img.save(cache, "JPEG", quality=quality, optimize=True)
    except Exception as e:
        # If scaling fails for any reason, fall back to full image.
        print(f"[scaled] failed for {character}/{filename}: {e}", file=sys.stderr)
        return send_file(src)

    resp = send_file(cache, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "max-age=86400"
    return resp


@app.get("/thumb/<character>/<path:filename>")
def serve_thumb(character: str, filename: str):
    return _serve_scaled(character, filename, THUMB_ROOT, THUMB_MAX_W, THUMB_QUALITY)


@app.get("/big/<character>/<path:filename>")
def serve_big(character: str, filename: str):
    """Modal/lightbox size for phones — full-res PNGs (~2.8MB) over Wi-Fi
    saturate the MT7922 radio (PC→AP→phone = every byte on air twice)."""
    return _serve_scaled(character, filename, BIG_ROOT, BIG_MAX_W, BIG_QUALITY)


@app.get("/api/download/<character>/<path:filename>")
def download_img(character: str, filename: str):
    path = OUTPUT_ROOT / character / filename
    if not path.is_file():
        return "not found", 404
    return send_file(path, as_attachment=True, download_name=filename)


@app.post("/api/mark/<character>/<name>")
def mark(character: str, name: str):
    body = request.get_json(force=True) or {}
    marks = load_marks(character)
    entry = marks.get(name, {})
    for k in ("keep", "reject", "hq", "note", "regen", "civitai", "stars"):
        if k in body:
            entry[k] = body[k]
    # clean up empty/zero
    entry = {k: v for k, v in entry.items() if v not in (False, None, "", 0)}
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
    # Regen-flagged notes get their own AUTO-REGEN section for claude to pick up.
    autoregen = sorted(k for k, v in marks.items() if v.get("regen") and v.get("note"))
    noted = sorted(
        k for k, v in marks.items()
        if v.get("note") and not v.get("keep") and not v.get("reject") and not v.get("regen")
    )
    rejected_noted = sorted(k for k, v in marks.items() if v.get("reject") and v.get("note"))
    lines = [
        f"# {character} — {len(keepers)} keepers, {len(autoregen)} auto-regen, {len(noted)} refine, {len(rejected_noted)} rejected w/ notes",
        "",
    ]
    if keepers:
        lines.append("## Keepers")
        for k in keepers:
            note = marks[k].get("note", "")
            hq = marks[k].get("hq", "")
            bits = []
            if hq:
                bits.append(f"hq={hq}")
            if marks[k].get("regen"):
                bits.append("REGEN")
            if note:
                bits.append(note)
            tail = f" — {', '.join(bits)}" if bits else ""
            lines.append(f"- {k}{tail}")
        lines.append("")
    if autoregen:
        lines.append("## Refine — AUTO-REGEN (execute immediately)")
        for k in autoregen:
            m = marks[k]
            kept = " [kept]" if m.get("keep") else ""
            lines.append(f"- {k}{kept} — {m['note']}")
        lines.append("")
    if noted:
        lines.append("## Refine (noted but not kept)")
        for k in noted:
            lines.append(f"- {k} — {marks[k]['note']}")
        lines.append("")
    if rejected_noted:
        lines.append("## Rejected — feedback on what went wrong")
        for k in rejected_noted:
            lines.append(f"- {k} — {marks[k]['note']}")
        lines.append("")
    three_star = sorted(k for k, v in marks.items() if v.get("stars") == 3)
    if three_star:
        lines.append("## ★★★ References")
        for k in three_star:
            note = marks[k].get("note", "")
            lines.append(f"- {k}" + (f" — {note}" if note else ""))
        lines.append("")
    if hq_groups:
        lines.append("## HQ Pass Queue")
        for hq_type, names in sorted(hq_groups.items()):
            lines.append(f"### {hq_type}")
            for n in sorted(names):
                lines.append(f"- {n}")
    return "\n".join(lines)


def build_submit_delta(character: str) -> str:
    """DELTA submit body: only keepers NEW / CHANGED / REMOVED since the last submit, + a count.
    Keeps the Claude-side sync payload tiny instead of re-dumping the whole keeper list every time.
    Snapshot of last-submitted keepers lives in data/<char>_lastsub.json."""
    import json as _json
    marks = load_marks(character)
    cur = {k: {"stars": v.get("stars", 0), "note": v.get("note", "")}
           for k, v in marks.items() if v.get("keep")}
    snap_path = DATA_DIR / f"{character}_lastsub.json"
    prev = {}
    if snap_path.is_file():
        try:
            prev = _json.loads(snap_path.read_text())
        except Exception:
            prev = {}
    new = [k for k in cur if k not in prev]
    changed = [k for k in cur if k in prev and cur[k] != prev[k]]
    removed = [k for k in prev if k not in cur]
    def line(k):
        s, n = cur[k]["stars"], cur[k]["note"]
        tail = (f" · {'★'*s}" if s else "") + (f" — {n}" if n else "")
        return f"- {k}{tail}"
    L = [f"# {character} — {len(cur)} keepers total · {len(new)} new · {len(changed)} changed · {len(removed)} removed since last submit", ""]
    if not (new or changed or removed):
        L.append("(no keeper changes since last submit)")
    if new:
        L.append("## NEW keepers"); L += [line(k) for k in sorted(new)]; L.append("")
    if changed:
        L.append("## CHANGED (star/note)")
        for k in sorted(changed):
            was = prev[k]
            L.append(line(k) + f"   [was {'★'*was.get('stars',0) or '0★'}{' / '+was['note'] if was.get('note') else ''}]")
        L.append("")
    if removed:
        L.append("## REMOVED (no longer keeper)"); L += [f"- {k}" for k in sorted(removed)]; L.append("")
    # new ★★★ this round are the high-signal ones — call them out
    new3 = [k for k in (new + changed) if cur[k]["stars"] == 3]
    if new3:
        L.append("## ★★★ (new/changed this round)"); L += [f"- {k}" for k in sorted(new3)]; L.append("")
    snap_path.write_text(_json.dumps(cur))
    return "\n".join(L)


@app.get("/api/export/<character>")
def export(character: str):
    return build_export(character), 200, {"Content-Type": "text/plain; charset=utf-8"}


def _log_learning(character, marks):
    """Append keep/reject verdict + prompt for each marked image to data/learning_log.jsonl.
    Deduped by (char, stem). Called from submit() BEFORE rejects are deleted so the signal survives."""
    import json as _json
    from PIL import Image as _Img
    log = DATA_DIR / "learning_log.jsonl"
    char_dir = OUTPUT_ROOT / character
    seen = set()
    if log.is_file():
        for ln in log.read_text().splitlines():
            try:
                o = _json.loads(ln); seen.add((o["char"], o["stem"]))
            except Exception:
                pass
    out = []
    for stem, m in marks.items():
        if not (m.get("keep") or m.get("reject")):
            continue
        if (character, stem) in seen:
            continue
        png = char_dir / f"{stem}.png"
        if not png.is_file():
            continue
        prompt = ""
        try:
            d = _json.loads(_Img.open(png).info.get("prompt", "{}"))
            for n in d.values():
                if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode":
                    t = n["inputs"].get("text", "")
                    if isinstance(t, str) and len(t) > 120 and "worst quality" not in t.lower():
                        prompt = t; break
        except Exception:
            pass
        out.append(_json.dumps({"char": character, "stem": stem, "keep": bool(m.get("keep")),
                                "stars": m.get("stars", 0), "prompt": prompt[:2000]}))
    if out:
        with open(log, "a") as f:
            f.write("\n".join(out) + "\n")


@app.post("/api/submit/<character>")
def submit(character: str):
    body = build_submit_delta(character)   # delta-only payload for the Claude sync hook
    path = DATA_DIR / f"{character}_submit.md"
    path.write_text(body)
    marks = load_marks(character)
    keepers = sum(1 for v in marks.values() if v.get("keep"))
    # LEARN-EVERY-RUN: log keep/reject verdict + prompt for each marked image to a growing JSONL,
    # BEFORE rejects are deleted below (so the reject signal survives). learn.py reads this.
    try:
        _log_learning(character, marks)
    except Exception as e:
        print("[learn] log skipped:", e)
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


import requests as http_requests


@app.get("/api/learn")
def api_learn():
    """Run learn.py and return its conversion report (what features over/under-convert)."""
    import subprocess
    dirs = [d for d in request.args.get("dirs", "").split(",") if d]
    cmd = ["python3", "/home/chremmler/claude/comfy2/learn.py"] + dirs
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        out = r.stdout or r.stderr or "(no output)"
    except Exception as e:
        out = f"learn failed: {e}"
    return jsonify({"report": out})


@app.get("/api/comfy_queue")
def comfy_queue():
    try:
        r = http_requests.get("http://127.0.0.1:8188/queue", timeout=2)
        d = r.json()
        running = len(d.get("queue_running", []))
        pending = len(d.get("queue_pending", []))
        return jsonify({"running": running, "pending": pending, "total": running + pending})
    except Exception:
        return jsonify({"running": 0, "pending": 0, "total": 0, "offline": True})


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from queue_sarah import BASE_WORKFLOW, COCK_MODES, queue_prompt
from queue_donna import _add_enhanced_nodes
import copy


def extract_prompt_from_png(png_path: Path) -> dict | None:
    """Read ComfyUI workflow from PNG metadata, return full peek + workflow."""
    try:
        img = Image.open(png_path)
        prompt_json = img.info.get("prompt")
        if not prompt_json:
            return None
        d = json.loads(prompt_json)
        pos = d.get("6", {}).get("inputs", {}).get("text", "")
        neg = d.get("7", {}).get("inputs", {}).get("text", "")
        k1 = d.get("3", {}).get("inputs", {})
        k2 = d.get("22", {}).get("inputs", {})
        seed = k1.get("seed", 0)
        cfg = k1.get("cfg")
        sampler = k1.get("sampler_name")
        scheduler = k1.get("scheduler")
        steps = k1.get("steps")
        cfg2 = k2.get("cfg") if k2 else None
        steps2 = k2.get("steps") if k2 else None
        ckpt = d.get("14", {}).get("inputs", {}).get("ckpt_name")
        # Detect cock mode from LoRA states
        lora_stack = d.get("17", {}).get("inputs", {})
        cock_mode = "erect"  # default
        active_loras = []
        if lora_stack:
            flac = lora_stack.get("lora_15", {})
            erect = lora_stack.get("lora_14", {})
            bulge = lora_stack.get("lora_16", {})
            if isinstance(flac, dict) and flac.get("on"):
                cock_mode = "flaccid"
            elif isinstance(bulge, dict) and bulge.get("on"):
                cock_mode = "bulge"
            elif isinstance(erect, dict) and erect.get("on"):
                cock_mode = "erect"
            # Collect all "on" LoRAs with their strengths
            for k, v in lora_stack.items():
                if k.startswith("lora_") and isinstance(v, dict) and v.get("on"):
                    name = v.get("lora", k)
                    # shorten filename to basename without .safetensors
                    if isinstance(name, str):
                        name = name.rsplit("/", 1)[-1].replace(".safetensors", "")
                    active_loras.append({
                        "slot": k,
                        "name": name,
                        "strength": v.get("strength", 0),
                    })
        # Detect PAG / NegPip nodes by class
        has_pag = False
        pag_scale = None
        pag_block = None
        has_negpip = False
        for nid, node in d.items():
            cls = node.get("class_type", "")
            if cls == "PerturbedAttention":
                has_pag = True
                inp = node.get("inputs", {})
                pag_scale = inp.get("scale")
                pag_block = f"{inp.get('unet_block','?')}/{inp.get('unet_block_id','?')}"
            elif cls == "CLIPNegPip":
                has_negpip = True
        return {
            "positive": pos,
            "negative": neg,
            "seed": seed,
            "cfg": cfg,
            "cfg2": cfg2,
            "sampler": sampler,
            "scheduler": scheduler,
            "steps": steps,
            "steps2": steps2,
            "checkpoint": ckpt.rsplit("/", 1)[-1] if isinstance(ckpt, str) else ckpt,
            "cock_mode": cock_mode,
            "loras": active_loras,
            "has_pag": has_pag,
            "pag_scale": pag_scale,
            "pag_block": pag_block,
            "has_negpip": has_negpip,
            "pipeline": (_pl := pipeline_detect.classify_workflow(d)).get("short", "?"),
            "pipeline_name": _pl.get("name", ""),
            "scene": _pl.get("scene", ""),
            "workflow": d,
        }
    except Exception as e:
        return None


@app.get("/api/peek/<character>/<name>")
def peek(character: str, name: str):
    """Workflow peek — returns parsed metadata (no full workflow dict)."""
    png_path = OUTPUT_ROOT / character / f"{name}.png"
    if not png_path.is_file():
        return jsonify({"error": "not found"}), 404
    info = extract_prompt_from_png(png_path)
    if not info:
        return jsonify({"error": "no metadata"}), 400
    # Strip heavy workflow dict from the response
    info.pop("workflow", None)
    return jsonify(info)


def rebuild_workflow(orig_workflow: dict, new_seed: int) -> dict:
    """Clone the original workflow with a fresh seed on every sampler node.

    Pipeline-agnostic: the primary KSampler is no longer always node "3"
    (DAMN/MoP put it at "23", Qwen/anima vary). Re-seed every node that
    carries a seed/noise_seed so "make similar" always varies the render.
    Each seed-bearing node gets a distinct derived seed.
    """
    wf = copy.deepcopy(orig_workflow)
    offset = 0
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for key in ("seed", "noise_seed"):
            if isinstance(inputs.get(key), int):
                inputs[key] = (new_seed + offset) % (2 ** 53)
                offset += 1
    if offset == 0 and "3" in wf:  # legacy fallback
        wf["3"].setdefault("inputs", {})["seed"] = new_seed
    return wf


@app.post("/api/make_similar/<character>/<name>")
def make_similar(character: str, name: str):
    """Queue new renders with same prompt, new seeds.
    ?pipeline=ril55_il converts to RIL55 IL-only on the fly.
    ?bss=0.0|0.5|1.5 sets BreastSag (only with pipeline=ril55_il).
    """
    png_path = OUTPUT_ROOT / character / f"{name}.png"
    if not png_path.is_file():
        return jsonify({"error": "image not found"}), 404
    info = extract_prompt_from_png(png_path)
    if not info:
        return jsonify({"error": "no prompt metadata in PNG"}), 400
    count = int(request.args.get("count", "2"))
    count = max(1, min(count, 8))
    convert_pipeline = request.args.get("pipeline", "")
    bss = float(request.args.get("bss", "0.0"))
    queued = []
    for _ in range(count):
        seed = random.randint(1, 2**53)
        wf = rebuild_workflow(info["workflow"], seed)
        if convert_pipeline == "ril55_il":
            wf = pipeline_detect.convert_to_ril55_il(wf, bss=bss)
        pid = queue_prompt(wf)
        queued.append({"seed": seed, "prompt_id": pid})
    return jsonify({"ok": True, "queued": queued, "cock_mode": info["cock_mode"],
                    "count": len(queued), "converted": convert_pipeline == "ril55_il"})


def _inject_pag(wf: dict, cfg: float = 5.0) -> dict:
    """Inject PerturbedAttention between model loader and KSamplers, drop primary CFG."""
    upstream_model = wf["3"]["inputs"].get("model", ["17", 0])
    wf["51"] = {
        "inputs": {
            "model": upstream_model,
            "scale": 3.0,
            "adaptive_scale": 0.0,
            "unet_block": "middle",
            "unet_block_id": 0,
            "sigma_start": -1.0,
            "sigma_end": -1.0,
            "rescale": 0.0,
            "rescale_mode": "full",
        },
        "class_type": "PerturbedAttention",
        "_meta": {"title": "Perturbed Attention"},
    }
    wf["3"]["inputs"]["model"] = ["51", 0]
    wf["3"]["inputs"]["cfg"] = cfg
    if "22" in wf:
        wf["22"]["inputs"]["model"] = ["51", 0]
        # 2nd-pass CFG already low — leave it
    return wf


@app.post("/api/redo_pag/<character>/<name>")
def redo_pag(character: str, name: str):
    """Queue a PAG re-render: inject PerturbedAttention @ scale 3.0, drop CFG→5, new seed."""
    png_path = OUTPUT_ROOT / character / f"{name}.png"
    if not png_path.is_file():
        return jsonify({"error": "image not found"}), 404
    info = extract_prompt_from_png(png_path)
    if not info:
        return jsonify({"error": "no prompt metadata in PNG"}), 400
    if info.get("has_pag"):
        return jsonify({"error": "already has PAG — use make_similar instead"}), 400
    wf = copy.deepcopy(info["workflow"])
    seed = random.randint(1, 2**53)
    _inject_pag(wf, cfg=5.0)
    wf["3"]["inputs"]["seed"] = seed
    if "22" in wf:
        wf["22"]["inputs"]["seed"] = seed + 1
    # Save into the character's own folder with a _pag suffix
    wf["9"]["inputs"]["filename_prefix"] = f"comfy/{character}/{character}_pag"
    pid = queue_prompt(wf)
    return jsonify({"ok": True, "prompt_id": pid, "seed": seed})


@app.get("/api/stars")
def stars_gallery():
    """Return all images with stars==n (default 3) across all characters, sorted by char then mtime.
    Pass ?n=1|2|3 to filter by star level. ?n=0 returns ALL starred (1-3)."""
    try:
        n = int(request.args.get("n", 3))
    except (TypeError, ValueError):
        n = 3
    if not OUTPUT_ROOT.exists():
        return jsonify([])
    result = []
    for p in sorted(OUTPUT_ROOT.iterdir()):
        if not p.is_dir():
            continue
        char = p.name
        marks = load_marks(char)
        hq_done = hq_rendered_set(char, p)
        for stem, entry in marks.items():
            s = entry.get("stars", 0)
            if (n == 0 and s >= 1) or (n != 0 and s == n):
                png = p / f"{stem}.png"
                if png.is_file():
                    st = png.stat()
                    pl = pipeline_detect.get_pipeline(char, stem, png)
                    result.append({
                        "character": char,
                        "name": stem,
                        "file": f"{stem}.png",
                        "mtime": st.st_mtime,
                        "keep": entry.get("keep", False),
                        "hq": entry.get("hq"),
                        "hq_done": stem in hq_done,
                        "note": entry.get("note", ""),
                        "stars": s,
                        "pipeline": pl.get("short", "?"),
                        "pipeline_name": pl.get("name", ""),
                    })
    result.sort(key=lambda x: (x["character"], x["mtime"]))
    pipeline_detect.save_cache()
    return jsonify(result)


@app.post("/api/import_keepers")
def import_keepers():
    """One-shot import of keeper stems from memory .md files into data/<char>.json.
    Idempotent — never clears existing marks.
    Returns {imported: N, skipped: N, chars: [...]} summary.
    """
    MEMORY_DIR = Path("/home/chremmler/.claude/projects/-home-chremmler-claude-comfy2/memory")
    total_imported = 0
    total_skipped = 0
    chars_touched = []

    for md_path in sorted(MEMORY_DIR.glob("*.md")):
        char = md_path.stem
        if any(char.startswith(pfx) for pfx in (
            "feedback_", "project_", "checkpoint_", "il_lora", "lora_",
            "duo_failures", "civitai", "bulge_", "MEMORY",
        )):
            continue
        char_dir = OUTPUT_ROOT / char
        if not char_dir.is_dir():
            continue

        text = md_path.read_text()
        marks = load_marks(char)
        changed = False

        for line in text.splitlines():
            stars_here = bool(re.search(r'[⭐★]{3}|\*{3}', line))
            for m in re.finditer(
                r'\b([a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]*?\d{4,5}_[a-zA-Z0-9_]*?)(?:\.png)?\b', line
            ):
                raw = m.group(1)
                stem_key = raw if raw.endswith("_") else raw + "_"
                png = char_dir / f"{stem_key}.png"
                if not png.is_file():
                    stem_key2 = raw.rstrip("_")
                    png = char_dir / f"{stem_key2}.png"
                    if png.is_file():
                        stem_key = stem_key2
                    else:
                        total_skipped += 1
                        continue
                entry = marks.get(stem_key, {})
                if not entry.get("keep"):
                    entry["keep"] = True
                    marks[stem_key] = entry
                    changed = True
                    total_imported += 1
                if stars_here and entry.get("stars", 0) < 3:
                    entry["stars"] = 3
                    marks[stem_key] = entry
                    changed = True

        if changed:
            save_marks(char, marks)
            chars_touched.append(char)

    return jsonify({"ok": True, "imported": total_imported, "skipped": total_skipped,
                    "chars": chars_touched})


@app.post("/api/queue_hq/<character>/<name>")
def queue_hq(character: str, name: str):
    """Queue an HQ pass (FaceDetailer + CockDetailer, no upscale) from PNG metadata."""
    png_path = OUTPUT_ROOT / character / f"{name}.png"
    if not png_path.is_file():
        return jsonify({"error": "image not found"}), 404
    info = extract_prompt_from_png(png_path)
    if not info:
        return jsonify({"error": "no prompt metadata in PNG"}), 400
    wf = copy.deepcopy(info["workflow"])
    m = re.search(r'_0*(\d+)_?$', name)
    num = m.group(1) if m else "00"
    wf["9"]["inputs"]["filename_prefix"] = f"comfy/{character}/{character}_hq_{num}"
    _add_enhanced_nodes(wf, upscale=False)
    pid = queue_prompt(wf)
    return jsonify({"ok": True, "prompt_id": pid, "num": num})


# ── Photoreal pass: 1.5x hires + CyberRealistic Pony img2img refine (the "creatil" upgrade) ──
INPUT_ROOT = Path("/home/chremmler/ComfyUI/input")
PR_REFINE_CKPT = "cyberrealisticPony_v180Coreshift.safetensors"
PR_POS = ("score_9, score_8_up, score_7_up, (photorealistic, raw photo, real photograph, real skin "
          "texture, visible skin pores, film grain:1.3), ")
PR_NEG = ("score_6, score_5, score_4, (anime, cartoon, illustration, cgi, 3d, render, doll, plastic skin, "
          "smooth airbrushed skin:1.3), worst quality, low quality, deformed, bad anatomy, bad hands, "
          "watermark, signature, (detached penis, floating penis:1.4), young, teen, child, ")


def _png_positive(png_path: Path) -> str:
    """Best-effort positive-prompt text from a ComfyUI PNG: follow the KSampler's positive link,
    else fall back to common node ids (20=creatil, 6=pony)."""
    try:
        d = json.loads(Image.open(png_path).info.get("prompt", "{}"))
    except Exception:
        return ""
    for node in d.values():
        if node.get("class_type") == "KSampler":
            link = node.get("inputs", {}).get("positive")
            if isinstance(link, list):
                t = d.get(link[0], {}).get("inputs", {}).get("text")
                if t:
                    return t
    for cand in ("20", "6"):
        t = d.get(cand, {}).get("inputs", {}).get("text")
        if t:
            return t
    return ""


@app.post("/api/photoreal/<character>/<name>")
def queue_photoreal(character: str, name: str):
    """Add the photoreal passes to one render: copy its pixels, img2img through CyberRealistic Pony
    at 1.5x with denoise 0.5. Output -> <char>_refine_<num> (picked up by the 'refined' badge)."""
    png_path = OUTPUT_ROOT / character / f"{name}.png"
    if not png_path.is_file():
        return jsonify({"error": "image not found"}), 404
    src_name = f"pr_{character}_{name}.png"
    try:
        shutil.copyfile(png_path, INPUT_ROOT / src_name)
    except Exception as e:
        return jsonify({"error": f"copy to input failed: {e}"}), 500
    m = re.search(r'_0*(\d+)_?$', name)
    num = m.group(1) if m else "00"
    pos = PR_POS + _png_positive(png_path)
    seed = random.randint(1, 2**53)
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": PR_REFINE_CKPT}},
        "2": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": -2}},
        "3": {"class_type": "LoadImage", "inputs": {"image": src_name}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}},
        "5": {"class_type": "LatentUpscaleBy", "inputs": {
            "samples": ["4", 0], "upscale_method": "bicubic", "scale_by": 1.5}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["2", 0]}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"text": PR_NEG, "clip": ["2", 0]}},
        "23": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["5", 0],
            "seed": seed, "steps": 28, "cfg": 5.0,
            "sampler_name": "dpmpp_2m_sde", "scheduler": "karras", "denoise": 0.45}},
        "24": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["1", 2]}},
        "25": {"class_type": "SaveImage", "inputs": {"images": ["24", 0],
                "filename_prefix": f"comfy/{character}/{character}_refine_{num}"}},
    }
    pid = queue_prompt(wf)
    return jsonify({"ok": True, "prompt_id": pid, "num": num})


# ── ComfyUI WebSocket proxy ──────────────────────────────────────────────────
# Runs as a background thread: connects server-side (no origin check) to
# ComfyUI's WebSocket and caches the latest preview frame + progress info.
# The browser then polls /api/comfy_preview instead of connecting directly.

_ws_lock = threading.Lock()
_ws_preview_b64: str | None = None   # latest JPEG preview as base64 string
_ws_progress: dict = {}              # {"step": N, "max": M}
_ws_rendering: bool = False          # True while a job is running


def _comfy_ws_proxy():
    global _ws_preview_b64, _ws_progress, _ws_rendering
    try:
        import websocket  # websocket-client
    except ImportError:
        return  # no library — silently skip proxy
    client_id = uuid.uuid4().hex
    ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
    while True:
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=_ws_on_message,
                on_error=lambda ws, err: None,
                on_close=lambda ws, *a: None,
            )
            ws.run_forever()
        except Exception:
            pass
        time.sleep(4)  # reconnect after disconnect


def _ws_on_message(ws, msg):
    global _ws_preview_b64, _ws_progress, _ws_rendering
    if isinstance(msg, bytes):
        # Binary frame: first 8 bytes = type header, rest = JPEG
        if len(msg) > 8:
            jpeg = msg[8:]
            b64 = base64.b64encode(jpeg).decode()
            with _ws_lock:
                _ws_preview_b64 = b64
                _ws_rendering = True
    else:
        try:
            data = json.loads(msg)
            if data.get("type") == "progress":
                with _ws_lock:
                    _ws_progress = {"step": data["data"]["value"], "max": data["data"]["max"]}
                    _ws_rendering = True
            elif data.get("type") == "status":
                q = data.get("data", {}).get("status", {}).get("exec_info", {}).get("queue_remaining", -1)
                if q == 0:
                    with _ws_lock:
                        _ws_rendering = False
        except Exception:
            pass


_proxy_thread = threading.Thread(target=_comfy_ws_proxy, daemon=True)
_proxy_thread.start()


@app.get("/api/comfy_preview")
def comfy_preview():
    """Return latest ComfyUI preview frame (base64 JPEG) + progress."""
    with _ws_lock:
        return jsonify({
            "rendering": _ws_rendering,
            "img": _ws_preview_b64,
            "step": _ws_progress.get("step"),
            "max": _ws_progress.get("max"),
        })


@app.get("/stars")
@app.get("/stars/<int:n>")
def stars_page(n=3):
    # one template, level-aware (reads star level from the URL path: /stars=3, /stars/2, /stars/1, /stars/0=all)
    resp = send_from_directory("static", "stars.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.get("/compare")
def compare_page():
    resp = send_from_directory("static", "compare.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.get("/api/compare/pairs/<character>")
def compare_pairs(character: str):
    """Return list of (original, hq) filename pairs for a character."""
    d = OUTPUT_ROOT / character
    if not d.is_dir():
        return jsonify([]), 404

    # Find all HQ files: <char>_hq_<num>_00001_.png
    hq_pattern = re.compile(r'^.+_hq_(\w+)_\d+_\.png$')
    orig_pattern = re.compile(r'^(.+?)_(\d+)_\.png$')

    # Build map: strip number -> hq filename
    hq_by_num: dict[str, str] = {}
    for f in d.glob("*.png"):
        m = hq_pattern.match(f.name)
        if m:
            # hq_num may be like "7" or "slutty_8"
            hq_by_num[m.group(1)] = f.name

    # Build map: num -> list of candidate original files
    candidates: dict[str, list[str]] = {}
    for f in sorted(d.glob("*.png")):
        if "_hq_" in f.name:
            continue
        m = orig_pattern.match(f.name)
        if not m:
            continue
        num_str = str(int(m.group(2)))
        candidates.setdefault(num_str, []).append(f.name)

    pairs = []
    for num_str, hq_name in sorted(hq_by_num.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        cands = candidates.get(num_str, [])
        if not cands:
            continue
        # Prefer file whose prefix matches character name
        preferred = next((c for c in cands if c.startswith(character + "_")), cands[0])
        pairs.append({"orig": preferred, "hq": hq_name, "num": num_str})

    return jsonify(pairs)


@app.get("/api/imgsize/<character>/<name>")
def imgsize(character: str, name: str):
    path = OUTPUT_ROOT / character / f"{name}.png"
    if not path.is_file():
        return jsonify({"error": "not found"}), 404
    try:
        img = Image.open(path)
        note = None
        raw = img.info.get("animation_note")
        if raw:
            try:
                note = json.loads(raw)
            except Exception:
                note = None
        return jsonify({"width": img.width, "height": img.height, "animation_note": note})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/animate")
def animate_page():
    resp = send_from_directory("static", "animate.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


import runpod_client


def _runpod_post(path: str, data: dict) -> dict:
    return runpod_client.post(path, data)


def _runpod_get(path: str) -> dict:
    return runpod_client.get(path)


def _download_video(job: dict, videos: list[dict]) -> None:
    """Download completed videos from RunPod to local VIDEO_DIR, with metadata sidecar."""
    char = job.get("character", "unknown")
    src_name = job.get("name", "unknown")
    dest_dir = VIDEO_DIR / char
    dest_dir.mkdir(parents=True, exist_ok=True)
    for v in videos:
        rel = v.get("rel", "")
        fname = rel.split("/")[-1]
        dest = dest_dir / (f"{src_name}__{fname}" if src_name else fname)
        if dest.exists():
            continue
        try:
            r = http_requests.get(v["url"], timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            # Save metadata sidecar
            meta = {
                "character": char, "src_name": src_name,
                "prompt": job.get("prompt", ""),
                "user_description": job.get("user_description", ""),
                "width": job.get("width", 0), "height": job.get("height", 0),
                "length": job.get("length", 81), "fps": job.get("fps", 24),
                "fast_mode": job.get("fast_mode", True),
                "downloaded": time.time(),
            }
            dest.with_suffix(".json").write_text(json.dumps(meta, indent=2))
            # Generate thumbnail
            thumb = DATA_DIR / "video_thumbs" / f"{char}__{dest.stem}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(dest), "-vframes", "1", "-q:v", "3", str(thumb)],
                capture_output=True, timeout=30,
            )
        except Exception:
            pass


# ── Auto-pull sweep ──────────────────────────────────────────────────────────
# Downloads ANY finished .mp4 on the ComfyUI instance that we don't already have,
# including clips dispatched outside keeperweb (standalone dispatch_*.py scripts).
# Complements the tracked-job poller (_bg_job_poll_loop, which handles Part 1).
def _prompt_from_history_entry(entry: dict) -> str:
    """Best-effort positive prompt from a /history entry's stored workflow."""
    try:
        workflow = entry.get("prompt", [None, None, {}])[2]
        texts = [n.get("inputs", {}).get("text", "") for n in workflow.values()
                 if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"]
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        positives = [t for t in texts if "low quality" not in t and "worst quality" not in t]
        return (positives or texts or [""])[0]
    except Exception:
        return ""


_AUTOPULL_SEEN = DATA_DIR / "autopull_seen.json"

def _seen_load() -> set:
    try:
        return set(json.loads(_AUTOPULL_SEEN.read_text()))
    except Exception:
        return set()

def _seen_save(s: set) -> None:
    try:
        _AUTOPULL_SEEN.write_text(json.dumps(sorted(s)))
    except Exception:
        pass


def _sweep_history() -> None:
    """Pull every finished .mp4 on the pod we have NEVER pulled before. Dedup via a
    PERSISTENT ledger of pulled ComfyUI filenames, so deleting a clip in keeper does
    NOT trigger a re-download."""
    history = _runpod_get("/history")
    seen = _seen_load()
    # Fold currently-present swept clips into the ledger so they're remembered even
    # if deleted later (covers clips pulled before this ledger existed).
    unc = VIDEO_DIR / "uncategorized"
    if unc.exists():
        for p in unc.glob("*.mp4"):
            seen.add(p.name)
    tracked = {p.name for p in VIDEO_DIR.rglob("*.mp4")}  # skip Part-1 files (src__fname)
    def have(fn):
        return fn in seen or any(n.endswith(fn) for n in tracked)
    changed = False
    for entry in history.values():
        if not isinstance(entry, dict):
            continue
        videos = []
        for node_output in entry.get("outputs", {}).values():
            for key in ("videos", "images"):
                for vinfo in node_output.get(key, []):
                    fname = vinfo.get("filename", "")
                    if not fname.endswith(".mp4") or have(fname):
                        continue
                    subfolder = vinfo.get("subfolder", "")
                    videos.append({
                        "rel": f"{subfolder}/{fname}".lstrip("/"),
                        "url": f"{RUNPOD_COMFY}/view?filename={fname}&subfolder={subfolder}&type=output",
                    })
                    seen.add(fname); changed = True  # remember permanently -> deletion won't re-pull
        if not videos:
            continue
        job = {"character": "uncategorized", "name": "",
               "prompt": _prompt_from_history_entry(entry),
               "user_description": "auto-swept (dispatched outside keeperweb)"}
        _download_video(job, videos)
    if changed or seen:
        _seen_save(seen)


def _autopull_sweep_loop() -> None:
    """Background thread: every 60s, sweep the pod for any finished clip we lack."""
    time.sleep(15)  # let startup + tracked-job poller settle first
    while True:
        try:
            _sweep_history()
        except Exception:
            pass
        time.sleep(60)


def _upload_to_runpod(image_path) -> str:
    """Upload image to RunPod ComfyUI input folder, return filename."""
    return runpod_client.upload_image(image_path)


def _wan_dims_for_image(png_path, base: int = 640, mult: int = 16,
                        lo: int = 320, hi: int = 1024) -> tuple[int, int]:
    """Derive WAN width/height from a source image's aspect ratio.

    Preserves the source aspect (no square distortion) while keeping pixel
    area ≈ base² so render time stays consistent. Snaps each side to a
    multiple of `mult` and clamps to [lo, hi]. Falls back to base×base if
    the image can't be read.
    """
    try:
        from PIL import Image
        w, h = Image.open(png_path).size
        ar = w / h
        th = (base * base / ar) ** 0.5
        tw = ar * th
        def snap(x):
            return max(lo, min(hi, int(round(x / mult)) * mult))
        return snap(tw), snap(th)
    except Exception:
        return base, base


@app.post("/api/animate")
def api_animate():
    """
    Body (JSON): { "character": "...", "name": "...", "prompt": "...",
                   "width": 640, "height": 640, "length": 81,
                   "fast_mode": true, "content_loras": [["Breast_Physics", 0.8]],
                   "engine": "wan"|"ltxv" }
    """
    from wan_workflow import build_wan_i2v_workflow
    body = request.get_json(force=True) or {}
    character = body.get("character", "")
    name = body.get("name", "")
    prompt = body.get("prompt", "")
    if not character or not name or not prompt:
        return jsonify({"error": "character, name and prompt required"}), 400

    png_path = OUTPUT_ROOT / character / f"{name}.png"
    if not png_path.is_file():
        return jsonify({"error": "image not found"}), 404

    engine = body.get("engine", "wan")
    fast_mode, quality_steps = _parse_mode(body.get("mode"), body.get("fast_mode", True))
    if engine == "ltxv":
        default_w, default_h, default_len = 768, 512, 97
    else:
        # Derive dims from source aspect so portrait images aren't squished into a square.
        default_w, default_h = _wan_dims_for_image(png_path)
        # 81 = one context window → stays faithful to source. Longer length crosses
        # into a 2nd window that hallucinates ("interprets") the image into CGI drift.
        # Duration is extended via RIFE playback (see wan_workflow.py) instead.
        default_len = 81
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "prompt_id": None,
        "character": character,
        "name": name,
        "user_description": prompt,
        "prompt": prompt,
        "engine": engine,
        "width": int(body.get("width", default_w)),
        "height": int(body.get("height", default_h)),
        "length": int(body.get("length", default_len)),
        "fps": int(body.get("fps", 24)),
        "fast_mode": fast_mode,
        "quality_steps": quality_steps,
        "content_loras": body.get("content_loras", []),
        # Partnered/POV overrides (kept man in frame, handheld POV, etc.) — used by
        # both the single dispatch path and svi_long. None => default behavior.
        "negative": body.get("negative") or None,
        "prefix_override": body.get("prefix_override", None),
        "out_stem": body.get("out_stem", None),
        "long_mode": bool(body.get("long_mode", False)),
        "target_clips": int(body.get("target_clips", 0)),
        "clip_prompts": body.get("clip_prompts", []),
        "clip_progress": "",
        "submitted": time.time(),
        "status": "pending",
        "videos": [],
    }
    with _animate_jobs_lock:
        _animate_jobs.insert(0, job)
    _save_jobs()
    return jsonify({"ok": True, "job_id": job_id, "status": "pending"})


@app.post("/api/animate_upload")
def api_animate_upload():
    """
    Multipart form: image file + fields (prompt, width, height, length, fps, fast_mode,
    content_loras, engine).
    Saves image to OUTPUT_ROOT/_uploads/, then queues an animate job.
    """
    from wan_workflow import build_wan_i2v_workflow
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    uploads_dir = OUTPUT_ROOT / "_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    saved_stem = request.form.get("_saved_stem", "").strip()
    if saved_stem:
        dest = uploads_dir / f"{saved_stem}.png"
        stem = saved_stem
        if not dest.is_file():
            saved_stem = ""  # fallback to re-upload

    if not saved_stem:
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "image file required"}), 400
        stem = re.sub(r"[^\w\-]", "_", Path(image_file.filename).stem if image_file.filename else "upload")[:60]
        dest = uploads_dir / f"{stem}.png"
        if dest.exists():
            stem = f"{stem}_{uuid.uuid4().hex[:6]}"
            dest = uploads_dir / f"{stem}.png"
        try:
            img = Image.open(image_file.stream).convert("RGB")
            img.save(dest, "PNG")
        except Exception as e:
            return jsonify({"error": f"could not save image: {e}"}), 400

    import json as _json
    try:
        content_loras = _json.loads(request.form.get("content_loras", "[]"))
    except Exception:
        content_loras = []

    engine = request.form.get("engine", "wan")
    default_w, default_h, default_len = (768, 512, 97) if engine == "ltxv" else (640, 640, 81)
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "prompt_id": None,
        "character": "_uploads",
        "name": stem,
        "user_description": prompt,
        "prompt": prompt,
        "engine": engine,
        "width": int(request.form.get("width", default_w)),
        "height": int(request.form.get("height", default_h)),
        "length": int(request.form.get("length", default_len)),
        "fps": int(request.form.get("fps", 24)),
        **dict(zip(("fast_mode", "quality_steps"), _parse_mode(request.form.get("mode"), request.form.get("fast_mode", "true").lower() != "false"))),
        "content_loras": content_loras,
        "submitted": time.time(),
        "status": "pending",
        "videos": [],
    }
    with _animate_jobs_lock:
        _animate_jobs.insert(0, job)
    _save_jobs()
    return jsonify({"ok": True, "job_id": job_id, "status": "pending", "character": "_uploads", "name": stem})


@app.get("/api/animate/status/<prompt_id>")
def api_animate_status(prompt_id: str):
    try:
        history = _runpod_get(f"/history/{prompt_id}")
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    if prompt_id not in history:
        try:
            queue = _runpod_get("/queue")
            running = [item[1] for item in queue.get("queue_running", [])]
            pending = [item[1] for item in queue.get("queue_pending", [])]
            if prompt_id in running:
                return jsonify({"status": "running"})
            if prompt_id in pending:
                return jsonify({"status": "queued"})
        except Exception:
            pass
        return jsonify({"status": "queued"})

    entry = history[prompt_id]
    if entry.get("status", {}).get("status_str") == "error":
        with _animate_jobs_lock:
            for job in _animate_jobs:
                if job["prompt_id"] == prompt_id:
                    job["status"] = "error"
                    break
        _save_jobs()
        return jsonify({"status": "error"})

    videos = []
    for node_output in entry.get("outputs", {}).values():
        for key in ("videos", "images"):
            for vinfo in node_output.get(key, []):
                fname = vinfo.get("filename", "")
                if not fname.endswith(".mp4"):
                    continue
                subfolder = vinfo.get("subfolder", "")
                rel = f"{subfolder}/{fname}".lstrip("/")
                videos.append({"rel": rel, "url": f"{RUNPOD_COMFY}/view?filename={fname}&subfolder={subfolder}&type=output"})

    completed = entry.get("status", {}).get("status_str") != "error"
    status = "done" if (completed and videos) else ("error" if not completed else "done")
    job_ref = None
    with _animate_jobs_lock:
        for job in _animate_jobs:
            if job["prompt_id"] == prompt_id:
                prev_status = job.get("status")
                job["status"] = status
                job["videos"] = videos
                if status == "done" and prev_status != "done":
                    job_ref = dict(job)
                break

    if job_ref and videos:
        threading.Thread(target=_download_video, args=(job_ref, videos), daemon=True).start()

    _save_jobs()
    return jsonify({"status": status, "videos": videos})


@app.post("/api/jobs/dispatch/<job_id>")
def api_dispatch(job_id: str):
    """Dispatch a pending job with a refined prompt to ComfyUI."""
    body = request.get_json(force=True) or {}
    with _animate_jobs_lock:
        job = next((j for j in _animate_jobs if j.get("job_id") == job_id), None)
    if not job:
        return jsonify({"error": "job not found"}), 404
    if job["status"] not in ("pending", "error"):
        return jsonify({"error": f"job already {job['status']}"}), 400

    refined_prompt = body.get("prompt", job["prompt"])
    job["prompt"] = refined_prompt

    if job.get("long_mode") and job.get("engine", "wan") == "wan":
        import svi_long
        if not job.get("target_clips") and not job.get("clip_prompts"):
            job["target_clips"] = svi_long.pick_clip_count(refined_prompt)
        job["status"] = "queued"
        _save_jobs()
        svi_long.start(
            job,
            output_root=OUTPUT_ROOT, video_dir=VIDEO_DIR,
            ensure_loras=_ensure_default_wan_loras,
            on_update=_save_jobs,
        )
        return jsonify({"ok": True, "long_mode": True,
                        "target_clips": job.get("target_clips"),
                        "clips": len(job.get("clip_prompts") or [])})

    png_path = OUTPUT_ROOT / job["character"] / f"{job['name']}.png"
    try:
        comfy_filename = _upload_to_runpod(png_path)
    except Exception as e:
        return jsonify({"error": f"upload failed: {e}"}), 502

    engine = job.get("engine", "wan")
    if engine == "ltxv":
        from ltxv_workflow import build_ltxv_i2v_workflow
        workflow = build_ltxv_i2v_workflow(
            image_filename=comfy_filename,
            positive_prompt=refined_prompt,
            width=job["width"], height=job["height"],
            length=job["length"], fps=job["fps"],
            filename_prefix="video/ltxv_ai",
        )
    else:
        from wan_workflow import build_wan_i2v_workflow
        # Per-job overrides for partnered/POV scenes: a custom negative (keep the
        # male partner in frame — default negative strips "other people"/"multiple
        # people") and a positive prefix override (e.g. allow handheld POV camera,
        # or drop the slow-motion / hands-away clause for rough sex).
        extra = {}
        if job.get("negative"):
            extra["negative_prompt"] = job["negative"]
        if job.get("prefix_override") is not None:
            extra["prefix_override"] = job["prefix_override"]
        workflow = build_wan_i2v_workflow(
            image_filename=comfy_filename,
            positive_prompt=refined_prompt,
            width=job["width"], height=job["height"],
            length=job["length"], fps=job["fps"],
            fast_mode=job["fast_mode"],
            quality_steps=job.get("quality_steps", 20),
            content_loras=_ensure_default_wan_loras(job.get("content_loras", [])),
            filename_prefix="video/wan_ai",
            use_rife=job.get("use_rife", True),  # RIFE 48fps smooth = reusable default (2026-06-11)
            **extra,
        )
    try:
        result = _runpod_post("/prompt", {"prompt": workflow})
    except Exception as e:
        return jsonify({"error": f"ComfyUI error: {e}"}), 502

    with _animate_jobs_lock:
        job["prompt_id"] = result.get("prompt_id")
        job["status"] = "queued"
    _save_jobs()
    return jsonify({"ok": True, "prompt_id": job["prompt_id"]})


@app.post("/api/rerun")
def api_rerun():
    """Create a pending job from a local video with feedback notes."""
    body = request.get_json(force=True) or {}
    character = body.get("character", "")
    src_name = body.get("src_name", "")
    original_prompt = body.get("original_prompt", "")
    feedback = body.get("feedback", "")
    width = int(body.get("width", 640))
    height = int(body.get("height", 640))
    length = int(body.get("length", 81))
    fast_mode, quality_steps = _parse_mode(body.get("mode"), body.get("fast_mode", True))
    if not character or not src_name:
        return jsonify({"error": "character and src_name required"}), 400
    # Build user_description combining original prompt + feedback
    user_desc = f"{original_prompt}\n\nFEEDBACK: {feedback}" if feedback else original_prompt
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id, "prompt_id": None,
        "character": character, "name": f"{src_name}_" if not src_name.endswith("_") else src_name,
        "user_description": user_desc,
        "prompt": original_prompt,
        "feedback": feedback,
        "width": width, "height": height, "length": length,
        "fps": 24, "fast_mode": fast_mode, "quality_steps": quality_steps, "content_loras": [],
        "submitted": time.time(), "status": "pending", "videos": [],
    }
    with _animate_jobs_lock:
        _animate_jobs.insert(0, job)
    return jsonify({"ok": True, "job_id": job_id})


@app.post("/api/plan_prompt")
def api_plan_prompt():
    body = request.get_json(force=True) or {}
    character = body.get("character", "")
    name = body.get("name", "")
    user_prompt = body.get("prompt", "")
    if not character or not name or not user_prompt:
        return jsonify({"error": "character, name and prompt required"}), 400
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500
    png_path = OUTPUT_ROOT / character / f"{name}.png"
    if not png_path.is_file():
        return jsonify({"error": "image not found"}), 404
    try:
        from planner import plan_workflow
        params = plan_workflow(png_path, user_prompt)
        return jsonify({"ok": True, "params": params})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/plan_upload")
def api_plan_upload():
    """
    Multipart: image file + user_prompt field.
    Analyzes image with Claude and returns suggested workflow params (does NOT queue a job).
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "image required"}), 400
    user_prompt = request.form.get("prompt", "").strip()
    if not user_prompt:
        return jsonify({"error": "prompt required"}), 400

    # Save to temp location for planner
    uploads_dir = OUTPUT_ROOT / "_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w\-]", "_", Path(image_file.filename).stem if image_file.filename else "upload")[:60]
    dest = uploads_dir / f"{stem}.png"
    if dest.exists():
        stem = f"{stem}_{uuid.uuid4().hex[:6]}"
        dest = uploads_dir / f"{stem}.png"
    try:
        img = Image.open(image_file.stream).convert("RGB")
        img.save(dest, "PNG")
    except Exception as e:
        return jsonify({"error": f"could not save image: {e}"}), 400

    try:
        from planner import plan_workflow
        params = plan_workflow(dest, user_prompt)
        params["_saved_stem"] = stem  # so submit can reuse without re-uploading
        return jsonify({"ok": True, "params": params})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/runpod_video/<path:filename>")
def proxy_runpod_video(filename: str):
    """Download video from RunPod to local cache and serve with range support."""
    import urllib.parse
    parts = filename.rsplit("/", 1)
    subfolder, fname = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
    cache_path = DATA_DIR / "video_cache" / filename.replace("/", "__")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        url = f"{RUNPOD_COMFY}/view?filename={urllib.parse.quote(fname)}&subfolder={urllib.parse.quote(subfolder)}&type=output"
        try:
            r = http_requests.get(url, timeout=60)
            r.raise_for_status()
            cache_path.write_bytes(r.content)
        except Exception as e:
            return f"proxy error: {e}", 502
    return send_file(cache_path, mimetype="video/mp4", conditional=True)


@app.get("/api/local_videos")
def api_local_videos():
    """List locally downloaded videos, newest first."""
    out = []
    if not VIDEO_DIR.exists():
        return jsonify([])
    for char_dir in sorted(VIDEO_DIR.iterdir()):
        if not char_dir.is_dir():
            continue
        for f in sorted(char_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime):
            parts = f.stem.split("__", 1)
            src_name = parts[0] if len(parts) == 2 else f.stem
            thumb = DATA_DIR / "video_thumbs" / f"{char_dir.name}__{f.stem}.jpg"
            meta_file = f.with_suffix(".json")
            try:
                meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            except Exception:
                meta = {}  # corrupt/empty sidecar (e.g. half-written on crash) -> don't break the page
            out.append({
                "character": char_dir.name,
                "src_name": src_name,
                "filename": f.name,
                "rel": f"{char_dir.name}/{f.name}",
                "mtime": f.stat().st_mtime,
                "size": f.stat().st_size,
                "has_thumb": thumb.exists(),
                "prompt": meta.get("prompt", ""),
                "user_description": meta.get("user_description", ""),
                "width": meta.get("width", 0),
                "height": meta.get("height", 0),
                "length": meta.get("length", 81),
                "fast_mode": meta.get("fast_mode", True),
                "keep": meta.get("keep", False),
                "reject": meta.get("reject", False),
                "note": meta.get("note", ""),
            })
    return jsonify(out)


@app.get("/local_video/<path:rel>")
def serve_local_video(rel: str):
    path = VIDEO_DIR / rel
    if not path.is_file():
        return "not found", 404
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.get("/video_thumb/<path:rel>")
def serve_video_thumb(rel: str):
    char, fname = rel.split("/", 1)
    thumb = DATA_DIR / "video_thumbs" / f"{char}__{fname.replace('.mp4', '.jpg')}"
    if not thumb.exists():
        # Generate on demand
        video = VIDEO_DIR / char / fname
        if video.is_file():
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video), "-vframes", "1", "-q:v", "3", str(thumb)],
                capture_output=True, timeout=30,
            )
    if not thumb.exists():
        return "no thumb", 404
    return send_file(thumb, mimetype="image/jpeg")


@app.post("/api/mark_video/<path:rel>")
def mark_video(rel: str):
    """Update keep/reject/note in the video's .json sidecar."""
    body = request.get_json(force=True) or {}
    video = VIDEO_DIR / rel
    if not video.is_file():
        return jsonify({"error": "not found"}), 404
    meta_file = video.with_suffix(".json")
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    for k in ("keep", "reject", "note"):
        if k in body:
            meta[k] = body[k]
    meta_file.write_text(json.dumps(meta, indent=2))
    return jsonify({"ok": True})


@app.post("/api/purge_rejected_videos")
def purge_rejected_videos():
    """Delete all video files (+ sidecar JSON + thumbnail) marked reject=true."""
    deleted = []
    for char_dir in VIDEO_DIR.iterdir():
        if not char_dir.is_dir():
            continue
        for meta_file in char_dir.glob("*.json"):
            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                continue
            if not meta.get("reject"):
                continue
            mp4 = meta_file.with_suffix(".mp4")
            thumb = DATA_DIR / "video_thumbs" / f"{char_dir.name}__{meta_file.stem}.jpg"
            for f in (mp4, meta_file, thumb):
                if f.exists():
                    f.unlink()
            deleted.append(str(mp4.name))
    return jsonify({"ok": True, "deleted": deleted, "count": len(deleted)})


@app.get("/videos")
def videos_page():
    resp = send_from_directory("static", "videos.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.get("/queue")
def queue_page():
    return send_from_directory("static", "queue.html")


@app.get("/api/jobs")
def api_jobs():
    with _animate_jobs_lock:
        return jsonify(list(_animate_jobs))


@app.patch("/api/jobs/update/<job_id>")
def api_jobs_update(job_id: str):
    data = request.get_json(force=True)
    found = False
    with _animate_jobs_lock:
        for job in _animate_jobs:
            if job["job_id"] == job_id and job["status"] == "pending":
                for key in ("prompt", "user_description"):
                    if key in data:
                        job[key] = data[key]
                found = True
                break
    if found:
        _save_jobs()
        return jsonify({"ok": True})
    return jsonify({"error": "job not found or not pending"}), 404


@app.delete("/api/jobs/cancel/<job_id>")
def api_jobs_cancel(job_id: str):
    found = False
    with _animate_jobs_lock:
        for job in _animate_jobs:
            if job["job_id"] == job_id and job["status"] == "pending":
                job["status"] = "cancelled"
                found = True
                break
    if found:
        _save_jobs()
        return jsonify({"ok": True})
    return jsonify({"error": "job not found or not pending"}), 404


def _terminate_runpod_pod(pod_id: str, api_key: str) -> bool:
    mutation = f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}'
    result = subprocess.run(
        ["curl", "-s", "-X", "POST",
         f"https://api.runpod.io/graphql?api_key={api_key}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"query": mutation})],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0


def _auto_shutdown_loop():
    """Terminate RunPod pod when all jobs done and ComfyUI queue is empty."""
    if not RUNPOD_POD_ID or not RUNPOD_API_KEY:
        return
    while True:
        time.sleep(60)
        try:
            with _animate_jobs_lock:
                jobs = list(_animate_jobs)
            if not jobs:
                continue
            all_done = all(j["status"] in ("done", "error") for j in jobs)
            if not all_done:
                continue
            # Check ComfyUI queue is empty
            try:
                queue = _runpod_get("/queue")
                running = len(queue.get("queue_running", []))
                pending = len(queue.get("queue_pending", []))
                if running > 0 or pending > 0:
                    continue
            except Exception:
                continue
            # All clear — terminate pod
            print(f"[auto-shutdown] All jobs done, terminating pod {RUNPOD_POD_ID}", flush=True)
            _terminate_runpod_pod(RUNPOD_POD_ID, RUNPOD_API_KEY)
            break
        except Exception:
            pass


def _bg_job_poll_loop():
    """Background thread: poll queued/running jobs every 20s and download completed videos."""
    time.sleep(10)  # let startup finish
    while True:
        try:
            with _animate_jobs_lock:
                active = [(j["job_id"], j["prompt_id"]) for j in _animate_jobs
                          if j["status"] in ("queued", "running") and j.get("prompt_id")]
            for job_id, prompt_id in active:
                try:
                    history = _runpod_get(f"/history/{prompt_id}")
                    if prompt_id not in history:
                        continue
                    entry = history[prompt_id]
                    if entry.get("status", {}).get("status_str") == "error":
                        with _animate_jobs_lock:
                            for job in _animate_jobs:
                                if job["prompt_id"] == prompt_id:
                                    job["status"] = "error"
                                    break
                        _save_jobs()
                        continue
                    videos = []
                    for node_output in entry.get("outputs", {}).values():
                        for key in ("videos", "images"):
                            for vinfo in node_output.get(key, []):
                                fname = vinfo.get("filename", "")
                                if not fname.endswith(".mp4"):
                                    continue
                                subfolder = vinfo.get("subfolder", "")
                                rel = f"{subfolder}/{fname}".lstrip("/")
                                videos.append({"rel": rel, "url": f"{RUNPOD_COMFY}/view?filename={fname}&subfolder={subfolder}&type=output"})
                    if not videos:
                        continue
                    job_ref = None
                    with _animate_jobs_lock:
                        for job in _animate_jobs:
                            if job["prompt_id"] == prompt_id and job["status"] != "done":
                                job["status"] = "done"
                                job["videos"] = videos
                                job_ref = dict(job)
                                break
                    if job_ref:
                        _save_jobs()
                        threading.Thread(target=_download_video, args=(job_ref, videos), daemon=True).start()
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(20)


_shutdown_thread = threading.Thread(target=_auto_shutdown_loop, daemon=True)
_shutdown_thread.start()

_bg_poll_thread = threading.Thread(target=_bg_job_poll_loop, daemon=True)
_bg_poll_thread.start()

# Part 2: sweep the pod for clips dispatched outside keeperweb (standalone scripts)
_sweep_thread = threading.Thread(target=_autopull_sweep_loop, daemon=True)
_sweep_thread.start()

# Load persisted jobs from previous session
with _animate_jobs_lock:
    _animate_jobs = _load_saved_jobs()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5151, debug=False, threaded=True)
