"""Thin RunPod ComfyUI HTTP helpers, shared by app.py and svi_long.py."""
import json
import os
import time
import requests

RUNPOD_COMFY = os.environ.get("RUNPOD_COMFY", "https://ff55ciault2yrs-8188.proxy.runpod.net")


def post(path: str, data: dict) -> dict:
    r = requests.post(f"{RUNPOD_COMFY}{path}", data=json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"}, timeout=30)
    r.raise_for_status()
    return r.json()


def get(path: str) -> dict:
    r = requests.get(f"{RUNPOD_COMFY}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def upload_image(image_path) -> str:
    suffix = str(image_path).lower()
    mt = "image/jpeg" if suffix.endswith((".jpg", ".jpeg")) else "image/png"
    with open(image_path, "rb") as f:
        r = requests.post(f"{RUNPOD_COMFY}/upload/image",
                          files={"image": (os.path.basename(str(image_path)), f, mt)}, timeout=60)
    r.raise_for_status()
    return r.json()["name"]


def download(url: str, dest) -> None:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)


def poll_until_done(prompt_id: str, timeout_s: int = 1800, interval_s: float = 4.0) -> list[dict]:
    """Block until the prompt finishes. Returns list of {rel, url} mp4 outputs.
    Raises RuntimeError on ComfyUI error or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hist = get(f"/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"ComfyUI error for {prompt_id}")
            videos = []
            for node_output in entry.get("outputs", {}).values():
                for key in ("videos", "images"):
                    for vinfo in node_output.get(key, []):
                        fn = vinfo.get("filename", "")
                        if not fn.endswith(".mp4"):
                            continue
                        sub = vinfo.get("subfolder", "")
                        videos.append({"rel": f"{sub}/{fn}".lstrip("/"),
                                       "url": f"{RUNPOD_COMFY}/view?filename={fn}&subfolder={sub}&type=output"})
            if videos:
                return videos
        time.sleep(interval_s)
    raise RuntimeError(f"timeout waiting for {prompt_id}")
