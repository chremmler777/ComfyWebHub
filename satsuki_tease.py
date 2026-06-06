"""satsuki teasing fan, lifts skirt to flaunt cock, 2-clip chain."""
import os, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"
import svi_long
from PIL import Image

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
VIDEO_DIR = "/home/chremmler/ComfyUI/output/videos"
CHAR, NAME = "satsuki", "satsuki_conv_lift_skirt_00001_"

B1 = ("She playfully grips the hem of her short white skirt and lifts it up to flaunt her thick erect "
      "cock to the viewer, teasing flirty smile, swaying her hips teasingly, her white angel wings "
      "flutter softly and her long black hair sways, the erect cock stays rigid and still hanging "
      "heavy with no independent jiggle, static camera, smooth playful continuous motion")
B2 = ("Still holding her short skirt lifted to show off her thick erect cock, she gives the viewer "
      "flirty pleading eyes and bites her lip, swaying gently, wings fluttering, hair swaying, the "
      "erect cock stays rigid and still, static camera, smooth playful continuous motion")

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
    "job_id": "satsuki_tease_1", "character": CHAR, "name": NAME,
    "prompt": "teasing fan lifting skirt flaunt cock",
    "clip_prompts": [B1, B2],
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
