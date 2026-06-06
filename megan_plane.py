"""megan plane teasing -> final cumshot, 3-clip chain."""
import os, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"
import svi_long
from PIL import Image

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
VIDEO_DIR = "/home/chremmler/ComfyUI/output/videos"
CHAR, NAME = "megan", "megan_plane_stroke_open_00006_"

B1 = ("Reclined in the airplane seat, she gazes up at the viewer with a sultry inviting look, slowly "
      "stroking her thick erect cock with her right hand in long lazy unhurried strokes, with her other "
      "hand she beckons the viewer closer patting her lap invitingly, her open white shirt shifts, the "
      "erect cock stays rigid and keeps its solid shape with no morphing, very slow sensual continuous "
      "stroking, static camera, no fast jerky movements")
B2 = ("Still reclined in the airplane seat, she strokes her thick erect cock faster and harder with her "
      "hand, her breathing getting heavier, sultry intense gaze at the viewer, building toward climax, "
      "the erect cock stays rigid and keeps its solid shape with no morphing, smooth rhythmic stroking, "
      "static camera")
B3 = ("Still reclined in the airplane seat, she strokes fast and thick white cum erupts and spurts from "
      "the tip of her cock, ropes of cum shooting up and landing across her chest and open white shirt, "
      "her mouth open in pleasure looking at the viewer, the erect cock stays rigid and solid, static camera")

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
    "job_id": "megan_plane_1", "character": CHAR, "name": NAME,
    "prompt": "plane teasing stroke to final cumshot",
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
