"""amara grab -> stroke -> squeeze+cum, 3-clip chain."""
import os, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"
import svi_long
from PIL import Image

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
VIDEO_DIR = "/home/chremmler/ComfyUI/output/videos"
CHAR, NAME = "amara", "amara_catwalk_naked_corset_00002_"

B1 = ("She lowers her hand and wraps her fingers firmly around the thick erect cock, gripping the "
      "shaft, the cock stays erect, smooth deliberate motion, static camera")
B2 = ("She strokes the thick erect cock up and down with her hand in a steady rhythm, continuous "
      "stroking motion along the shaft, the cock stays erect, static camera")
B3 = ("She squeezes the thick erect cock hard with her hand and thick white cum spurts out from the "
      "tip, ropes of cum shooting out as she squeezes, the cock stays erect, static camera")

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
    "job_id": "amara_squeeze_1", "character": CHAR, "name": NAME,
    "prompt": "grab stroke squeeze cum",
    "clip_prompts": [B1, B2, B3],
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
