"""satsuki refined: slow seductive talking invitation, 3-clip chain."""
import os, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"
import svi_long
from PIL import Image

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
VIDEO_DIR = "/home/chremmler/ComfyUI/output/videos"
CHAR, NAME = "satsuki", "satsuki_conv_lift_skirt_00001_"

B1 = ("She looks directly at the viewer and talks slowly and seductively, her lips moving gently and "
      "sensually as she speaks, soft seductive smile, slight slow head tilt, her white angel wings "
      "flutter softly and her long black hair sways gently, the erect cock stays rigid and still "
      "hanging heavy with no jiggle, very slow deliberate smooth motion, no abrupt movements, static camera")
B2 = ("Still talking slowly and seductively, she slowly lifts the hem of her short white skirt to reveal "
      "her thick erect cock to the viewer, sensual inviting expression, slow deliberate motion, wings "
      "fluttering softly, the erect cock stays rigid and still, very slow smooth motion, no abrupt "
      "movements, static camera")
B3 = ("She gives the viewer flirty seductive bedroom eyes and slowly curls her finger in a come-closer "
      "beckon, mouthing a seductive invitation to come closer, biting her lip sensually, very slow smooth "
      "deliberate motion, wings fluttering softly, the erect cock stays rigid and still, no abrupt "
      "movements, static camera")

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
    "job_id": "satsuki_invite_1", "character": CHAR, "name": NAME,
    "prompt": "slow seductive talking invitation come closer",
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
