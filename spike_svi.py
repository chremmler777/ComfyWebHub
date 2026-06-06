"""One-off SVI long-video spike: 3-clip chain on cw_s1_00002_."""
import os, sys, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"

from PIL import Image
import svi_long

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
VIDEO_DIR = "/home/chremmler/ComfyUI/output/videos"

CHAR, NAME = "cockworship", "cw_s1_00002_"
img = Image.open(os.path.join(OUTPUT_ROOT, CHAR, f"{NAME}.png"))
W, H = img.size
print(f"source {NAME} {W}x{H} -> svi_dims {svi_long.svi_dims(W, H)}", flush=True)

# Per-clip beats, refined per WAN conventions (slow, deliberate, continuous,
# static camera, penis-state reinforced for SmoothFutanaris).
BEAT_GRAB = ("She slowly raises one hand and wraps her fingers firmly around the thick erect cock, "
             "gripping the shaft, slow deliberate motion, static camera, the cock stays erect pointing up, "
             "smooth natural movement, no fast movements")
BEAT_STROKE = ("She strokes the thick erect cock up and down with her hand in a slow steady rhythm, "
               "continuous stroking motion, fingers sliding along the shaft, the cock stays erect pointing up, "
               "languid pace, static camera, smooth natural movement")
BEAT_CUM = ("She keeps stroking as thick white cum spurts from the tip of the cock onto her face, "
            "ropes of cum landing across her cheeks and open mouth and tongue, she keeps her tongue out, "
            "slow motion, static camera, the cock stays erect pointing up")

DEFAULT_WAN_LORAS = [("SmoothFutanaris", 0.7)]
def ensure_loras(content_loras):
    merged = [list(x) for x in (content_loras or [])]
    have = {x[0] for x in merged}
    for name, s in DEFAULT_WAN_LORAS:
        if name not in have:
            merged.append([name, s])
    return merged

job = {
    "job_id": "spike_svi_1",
    "character": CHAR, "name": NAME,
    "prompt": "self-stroke to facial",
    "clip_prompts": [BEAT_GRAB, BEAT_STROKE, BEAT_CUM],
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
