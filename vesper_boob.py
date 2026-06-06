"""vesper boob dance -> jiggle -> cum soaking top, 3-clip chain."""
import os, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"
import svi_long
from PIL import Image

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
VIDEO_DIR = "/home/chremmler/ComfyUI/output/videos"
CHAR, NAME = "vesper", "vesper_gym_locker_room_00004_"

B1 = ("Her enormous breasts bounce up and down in a rhythmic dance under her tight crop top, the "
      "fabric of the top stretches and deforms as her thick erect penis pushes and pokes up against "
      "it from below, the penis stays erect pointing up, smooth rhythmic bouncing, she smiles, static camera")
B2 = ("Her enormous breasts jiggle and bounce repeatedly under the tight top, the top deforms where "
      "her erect penis pokes against it, playful bouncing motion, the penis stays erect pointing up, "
      "she smiles, static camera")
B3 = ("Thick white cum spurts up from her erect penis onto her top, soaking and wetting the fabric of "
      "her crop top with cum, her huge breasts still jiggling, she smiles with pleasure, static camera")

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
    "job_id": "vesper_boob_1", "character": CHAR, "name": NAME,
    "prompt": "boob bounce penis poke then cum on top",
    "clip_prompts": [B1, B2, B3],
    "width": W, "height": H, "fps": 24,
    "content_loras": [["Bouncing_Boobs", 0.8]],
}

def on_update():
    print(f"[{time.strftime('%H:%M:%S')}] status={job.get('status')} "
          f"progress={job.get('clip_progress')} err={job.get('error')}", flush=True)

t0 = time.time()
svi_long.run_long_job(job, output_root=OUTPUT_ROOT, video_dir=VIDEO_DIR,
                      ensure_loras=ensure_loras, on_update=on_update)
print(f"DONE in {time.time()-t0:.0f}s status={job.get('status')} videos={job.get('videos')} err={job.get('error')}", flush=True)
