"""ss_00080 anal riding + bouncing boobs + flowing cum, 2-clip chain."""
import os, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"
import svi_long
from PIL import Image

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
VIDEO_DIR = "/home/chremmler/ComfyUI/output/videos"
CHAR, NAME = "shemalesurprise", "ss_00080_"

RIDE = ("POV she rides up and down on the cock in a continuous anal riding motion, bouncing "
        "rhythmically, her large breasts bounce and jiggle with each motion, her thick erect cock "
        "bounces and stays erect pointing up, thick white semen keeps spurting and flowing from her "
        "cock across her chest, she moans with mouth open, smooth rhythmic bouncing, static camera")

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
    "job_id": "ss_ride_1", "character": CHAR, "name": NAME,
    "prompt": "anal riding bouncing boobs flowing cum",
    "clip_prompts": [RIDE, RIDE],
    "width": W, "height": H, "fps": 24,
    "content_loras": [["Anal_Sex", 0.8], ["Bouncing_Boobs", 0.8], ["CumShot", 0.8]],
}

def on_update():
    print(f"[{time.strftime('%H:%M:%S')}] status={job.get('status')} "
          f"progress={job.get('clip_progress')} err={job.get('error')}", flush=True)

t0 = time.time()
svi_long.run_long_job(job, output_root=OUTPUT_ROOT, video_dir=VIDEO_DIR,
                      ensure_loras=ensure_loras, on_update=on_update)
print(f"DONE in {time.time()-t0:.0f}s status={job.get('status')} videos={job.get('videos')} err={job.get('error')}", flush=True)
