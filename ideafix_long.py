"""ideafix_127279336 longer stroke loop, 4-clip chain (reuses prior prompt)."""
import os, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"
import svi_long
from PIL import Image

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
VIDEO_DIR = "/home/chremmler/ComfyUI/output/videos"
CHAR, NAME = "ideas", "ideafix_127279336_00001_"

STROKE = ("A tattooed futa woman with purple hair and a black cap sits on a wooden table under a warm "
          "hanging lamp, an open navy jacket framing her bare tattooed breasts. She slowly strokes her "
          "cock with one hand in a steady rhythm, her breasts swaying softly, her other hand resting on "
          "her thigh, dark-lipped smile holding the camera. Continuous slow looping motion, sultry and "
          "relaxed, stable smooth cock.")

DEFAULT_WAN_LORAS = [("SmoothFutanaris", 0.7)]
def ensure_loras(content_loras):
    merged = [list(x) for x in (content_loras or [])]
    have = {x[0] for x in merged}
    for name, s in DEFAULT_WAN_LORAS:
        if name not in have:
            merged.append([name, s])
    return merged

W, H = Image.open(f"{OUTPUT_ROOT}/{CHAR}/{NAME}.png").size
print(f"{NAME} {W}x{H} -> {svi_long.svi_dims(W,H)}", flush=True)

job = {
    "job_id": "ideafix_long_1", "character": CHAR, "name": NAME,
    "prompt": "longer slow stroke loop",
    "clip_prompts": [STROKE, STROKE, STROKE, STROKE],   # 4 clips ~13s
    "width": W, "height": H, "fps": 24,
    "content_loras": [],
}

def on_update():
    print(f"[{time.strftime('%H:%M:%S')}] status={job.get('status')} "
          f"progress={job.get('clip_progress')} err={job.get('error')}", flush=True)

t0 = time.time()
svi_long.run_long_job(job, output_root=OUTPUT_ROOT, video_dir=VIDEO_DIR,
                      ensure_loras=ensure_loras, on_update=on_update)
print(f"DONE in {time.time()-t0:.0f}s status={job.get('status')} videos={job.get('videos')} err={job.get('error')}", flush=True)
