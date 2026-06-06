#!/usr/bin/env python3
"""Build a long VACE video by chaining reference-anchored 81-frame chunks.

Each chunk:
  - uses the SAME reference image  -> identity held across the whole video
  - chunk 0: anchored to the reference (frame 0 = source)
  - chunk N: first OVERLAP frames = tail of previous chunk -> seamless motion carry
Segments are concatenated (dropping the duplicated overlap) into one long clip.

VRAM stays at one 81-frame chunk regardless of total length.

Pod has no ffmpeg, so frame extraction is done locally and the overlap clip is
scp'd into the pod's ComfyUI/input dir for the next chunk's VHS_LoadVideo.

Usage:
  python3 vace_long.py <source_png> <out_name>
Prompts/chunk count are configured in MAIN below.
"""
import json, subprocess, sys, time, os

POD = "https://1989lfefs04uky-8188.proxy.runpod.net"
SSH = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
       "-p", "38218", "root@195.26.233.98", "-i", os.path.expanduser("~/.ssh/id_ed25519")]
SCP = ["scp", "-o", "StrictHostKeyChecking=no", "-P", "38218",
       "-i", os.path.expanduser("~/.ssh/id_ed25519")]
POD_INPUT = "/workspace/ComfyUI/input"
OVERLAP = 16
LENGTH = 81
FPS = 24

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vace_workflow import build_vace_workflow


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def curl_json(path, data=None, method="GET"):
    cmd = ["curl", "-s", "-m", "60", f"{POD}{path}"]
    if data is not None:
        cmd += ["-X", "POST", "-H", "Content-Type: application/json", "--data", json.dumps(data)]
    r = sh(cmd)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"_raw": r.stdout[:300]}


def upload_image(local_path):
    r = sh(["curl", "-s", "-m", "120", "-X", "POST", f"{POD}/upload/image",
            "-F", f"image=@{local_path}", "-F", "overwrite=true"])
    return json.loads(r.stdout)["name"]


def poll_done(pid, label, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = curl_json(f"/history/{pid}")
        if h and pid in h:
            outs = h[pid].get("outputs", {})
            for n in outs.values():
                for k in ("images", "gifs", "videos"):
                    for f in n.get(k, []):
                        if f.get("filename", "").endswith(".mp4"):
                            print(f"  [{label}] done -> {f['filename']}")
                            return f["filename"], f.get("subfolder", "")
        time.sleep(8)
    raise TimeoutError(f"{label} timed out")


def download(filename, subfolder, dest):
    sh(["curl", "-s", "-m", "180",
        f"{POD}/view?filename={filename}&subfolder={subfolder}&type=output", "-o", dest])
    return dest


def render_chunk(ref_name, prompt, idx, keep_video=None, n_keep=1, seed=12345):
    wf = build_vace_workflow(
        reference_image=ref_name, positive_prompt=prompt,
        width=480, height=832, length=LENGTH, fps=FPS, seed=seed + idx,
        filename_prefix=f"video/vace_seg{idx}",
        keep_video=keep_video, n_keep=n_keep,
    )
    resp = curl_json("/prompt", {"prompt": wf})
    pid = resp.get("prompt_id")
    if not pid:
        raise RuntimeError(f"chunk {idx} not queued: {resp}")
    print(f"  [chunk {idx}] queued {pid} (keep_video={keep_video}, n_keep={n_keep})")
    return poll_done(pid, f"chunk{idx}")


def main(src_png, out_name, prompts):
    print("uploading reference image...")
    ref = upload_image(src_png)
    print("  ref:", ref)
    segs = []
    keep_video, n_keep = None, 1
    for i, prompt in enumerate(prompts):
        fn, sub = render_chunk(ref, prompt, i, keep_video, n_keep)
        local = f"/tmp/vlong_seg{i}.mp4"
        download(fn, sub, local)
        segs.append(local)
        if i < len(prompts) - 1:
            # extract last OVERLAP frames -> overlap clip for next chunk
            ov = f"/tmp/vlong_overlap{i+1}.mp4"
            sh(["ffmpeg", "-y", "-sseof", f"-{OVERLAP/FPS+0.2:.2f}", "-i", local,
                "-frames:v", str(OVERLAP), "-r", str(FPS), "-an", ov])
            # scp into pod input dir
            ovname = f"vlong_overlap{i+1}.mp4"
            sh(SCP + [ov, f"root@195.26.233.98:{POD_INPUT}/{ovname}"])
            keep_video, n_keep = ovname, OVERLAP
            print(f"  overlap -> {ovname}")
    # concat: seg0 full + seg_i[OVERLAP:]
    print("concatenating...")
    parts = []
    for i, s in enumerate(segs):
        if i == 0:
            parts.append(s)
        else:
            trimmed = f"/tmp/vlong_trim{i}.mp4"
            sh(["ffmpeg", "-y", "-i", s, "-vf", f"select=gte(n\\,{OVERLAP}),setpts=N/{FPS}/TB",
                "-r", str(FPS), "-an", trimmed])
            parts.append(trimmed)
    listf = "/tmp/vlong_list.txt"
    with open(listf, "w") as fh:
        for p in parts:
            fh.write(f"file '{p}'\n")
    out_path = f"/home/chremmler/ComfyUI/output/videos/_vace_tests/{out_name}.mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an", out_path])
    # report
    pr = sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", out_path])
    print(f"DONE -> {out_path}  frames={pr.stdout.strip()}")
    return out_path


if __name__ == "__main__":
    src = sys.argv[1]
    out = sys.argv[2]
    PROMPTS = [
        "Her soft cock hangs heavy as her hand playfully teases and lifts it, fingers stroking and playing with the shaft, the cock slowly starting to stiffen. Static camera, smooth slow motion.",
        "Her cock is now fully erect and hard, and her hand strokes it firmly up and down the shaft in a steady building rhythm, fist gliding along the length. The erect cock stays rigid, no morphing. Static camera, smooth rhythmic motion.",
        "She strokes her hard cock faster and then it erupts, thick white cum spurting out from the tip in rhythmic surges, balls pulsing with each surge. The cock stays rigid. Static camera, smooth motion.",
    ]
    main(src, out, PROMPTS)
