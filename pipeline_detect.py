"""Pipeline detection and conversion utilities for keeperweb."""
import copy
import json
import threading
from pathlib import Path

from PIL import Image

PONY_STAGE_NODES = ("14", "17", "6", "7", "22", "8", "9", "104", "105")
RIL55 = "realismIllustriousBy_v55FP16.safetensors"
ILUSTREAL_V5 = "ilustreal_v50VAE.safetensors"
LUSTIFY = "lustify"  # lustifySDXLNSFW_apexV8 — the SHEMALE pipeline base
IL_QUALITY_HEADER = (
    "masterpiece, best quality, very aesthetic, photorealistic, "
    "99bsy99, absurdres, highres, very awa,"
)


def classify_workflow(wf: dict) -> dict:
    """Classify a workflow dict. Returns {name, short, scene}."""
    il_ckpt = wf.get("100", {}).get("inputs", {}).get("ckpt_name", "")
    pony_ckpt = wf.get("14", {}).get("inputs", {}).get("ckpt_name", "")
    has_pony_stage = any(n in wf for n in ("17", "22"))

    prefix = (
        wf.get("990", {}).get("inputs", {}).get("filename_prefix")  # inline-detailer save
        or wf.get("23", {}).get("inputs", {}).get("filename_prefix")
        or wf.get("9", {}).get("inputs", {}).get("filename_prefix")
        or ""
    )
    scene = prefix.rsplit("/", 1)[-1] if "/" in prefix else prefix

    if il_ckpt and not has_pony_stage:
        if RIL55 in il_ckpt:
            name, short = "RIL55 IL-only", "RIL55"
        elif ILUSTREAL_V5 in il_ckpt:
            name, short = "Ilustreal v5 IL-only", "IL-v5"
        elif LUSTIFY in il_ckpt.lower():
            name, short = "Shemale (Lustify)", "shemalelustpon"
        else:
            base = il_ckpt.rsplit("/", 1)[-1].replace(".safetensors", "")[:16]
            name, short = f"IL-only ({base})", "IL"
    elif il_ckpt and has_pony_stage:
        if RIL55 in il_ckpt:
            name, short = "RIL55 → Pony", "RIL55+P"
        else:
            name, short = "Ilustreal → Pony", "IL+P"
    elif pony_ckpt:
        if "v17" in pony_ckpt or "v180" in pony_ckpt:
            name, short = "Pony v17", "Pony17"
        elif "v16" in pony_ckpt or "v160" in pony_ckpt:
            name, short = "Pony v16", "Pony16"
        else:
            base = pony_ckpt.rsplit("/", 1)[-1].replace(".safetensors", "")[:16]
            name, short = f"Pony ({base})", "Pony"
    else:
        name, short = "Unknown", "?"

    return {"name": name, "short": short, "scene": scene}


def read_pipeline_from_png(png_path: Path) -> dict | None:
    try:
        img = Image.open(png_path)
        raw = img.info.get("prompt")
        if not raw:
            return None
        wf = json.loads(raw)
        return classify_workflow(wf)
    except Exception:
        return None


# ── In-process cache ────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()
_cache_path: Path | None = None
_cache_dirty = False


def init_cache(data_dir: Path):
    global _cache, _cache_path
    _cache_path = data_dir / "pipeline_cache.json"
    if _cache_path.exists():
        try:
            _cache = json.loads(_cache_path.read_text())
        except Exception:
            _cache = {}


def _save_cache():
    global _cache_dirty
    if not _cache_path or not _cache_dirty:
        return
    try:
        with _cache_lock:
            _cache_path.write_text(json.dumps(_cache))
            _cache_dirty = False
    except Exception:
        pass


def get_pipeline(char: str, stem: str, png_path: Path) -> dict:
    """Return pipeline info for one image; build from PNG + cache."""
    key = f"{char}/{stem}"
    mtime = png_path.stat().st_mtime
    with _cache_lock:
        entry = _cache.get(key)
    if entry and abs(entry.get("mtime", 0) - mtime) < 1:
        return entry["pipeline"]
    info = read_pipeline_from_png(png_path) or {"name": "Unknown", "short": "?", "scene": stem}
    with _cache_lock:
        global _cache_dirty
        _cache[key] = {"mtime": mtime, "pipeline": info}
        _cache_dirty = True
    # Persist lazily — caller should call save_cache() at end of request
    return info


def save_cache():
    _save_cache()


# ── Workflow conversion ──────────────────────────────────────────────────────

def _find_il_chain_tail(wf: dict) -> str | None:
    """Last LoraLoader node ID in IL chain (excluding testicle nodes 130/131)."""
    lora_nodes = [
        nid for nid, n in wf.items()
        if n.get("class_type") == "LoraLoader" and nid not in ("130", "131")
    ]
    if not lora_nodes:
        return None
    try:
        return str(max(int(n) for n in lora_nodes))
    except ValueError:
        return lora_nodes[-1]


def convert_to_ril55_il(wf: dict, bss: float = 0.0) -> dict:
    """
    Convert an IL-based workflow to RIL55 IL-only:
      - Swap IL ckpt to RIL55
      - Strip Pony stage nodes
      - CFG → 4.5
      - BSS (node 116) → bss
      - Inject 99bsy99 quality header into IL positive if missing
      - Add testicle LoRAs (nodes 130/131) if absent
    Works on already-IL workflows (RIL55 or Ilustreal); does NOT convert
    Pony-only workflows (those lack the IL LoRA chain nodes).
    """
    wf = copy.deepcopy(wf)

    if "100" not in wf:
        return wf  # can't convert Pony-only without full IL chain

    wf["100"]["inputs"]["ckpt_name"] = RIL55

    # Strip Pony stage
    for n in PONY_STAGE_NODES:
        wf.pop(n, None)

    # CFG 4.5
    if "3" in wf:
        wf["3"]["inputs"]["cfg"] = 4.5

    # BSS
    if "116" in wf:
        wf["116"]["inputs"]["strength_model"] = bss
        wf["116"]["inputs"]["strength_clip"] = bss

    # Inject quality header into IL positive text (node 115 → CLIPTextEncode)
    if "115" in wf:
        text = wf["115"]["inputs"].get("text", "")
        if "99bsy99" not in text:
            wf["115"]["inputs"]["text"] = IL_QUALITY_HEADER + " " + text

    # Testicle LoRAs — only add if not already present
    if "130" not in wf:
        tail = _find_il_chain_tail(wf)
        if tail:
            wf["130"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": [tail, 0], "clip": [tail, 1],
                    "lora_name": "Illustrious/Testicle_Size_Slider_IL.safetensors",
                    "strength_model": 1.5, "strength_clip": 1.5,
                },
            }
            wf["131"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["130", 0], "clip": ["130", 1],
                    "lora_name": "Illustrious/Sagging_Testicles_IL.safetensors",
                    "strength_model": 0.5, "strength_clip": 0.5,
                },
            }
            if "3" in wf:
                wf["3"]["inputs"]["model"] = ["131", 0]
            if "115" in wf:
                wf["115"]["inputs"]["clip"] = ["131", 1]

    return wf
