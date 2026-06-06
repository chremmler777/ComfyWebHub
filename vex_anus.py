"""vex rolls back and shows anus, 2-clip chain."""
import os, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"
import svi_long
from PIL import Image

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
VIDEO_DIR = "/home/chremmler/ComfyUI/output/videos"
CHAR, NAME = "vex", "vex_piledriver_00014_"

B1 = ("She rolls backward and lifts her hips and legs up over her head into a piledriver position, "
      "tilting her pelvis up toward the viewer, smooth deliberate rolling motion, the thick erect cock "
      "stays rigid and still, she smiles at the camera, static camera, no abrupt movements")
B2 = ("Now rolled back with her hips raised high, she spreads her legs wide and presents her anus to the "
      "viewer, showing off her asshole, slight teasing hip movement, the thick erect cock stays rigid and "
      "still, she smiles, smooth slow motion, static camera, no abrupt movements")

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
    "job_id": "vex_anus_1", "character": CHAR, "name": NAME,
    "prompt": "rolling back and showing anus",
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
