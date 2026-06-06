"""ideafix_65425222 refined: slow seductive talking invitation, 4-clip chain."""
import os, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"
import svi_long
from PIL import Image

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
VIDEO_DIR = "/home/chremmler/ComfyUI/output/videos"
CHAR, NAME = "ideas", "ideafix_65425222_00005_"

B1 = ("She looks directly at the viewer and talks very slowly and seductively, her dark lips moving "
      "gently and sensually as she speaks, soft seductive smile, slight slow head movement, one hand "
      "resting on the thick erect cock, the cock stays erect pointing up, very slow deliberate smooth "
      "motion, no abrupt movements, static camera")
B2 = ("Still talking slowly and seductively, she slowly begins stroking the thick erect cock up and down "
      "with her hand in a gentle unhurried rhythm, holding the viewer with a sultry gaze, the cock stays "
      "erect pointing up, very slow smooth motion, no abrupt movements, static camera")
B3 = ("She slowly raises her hand and points at the viewer, then curls her finger in a slow seductive "
      "come-closer beckon, inviting the viewer in, sensual bedroom eyes, the cock stays erect pointing up, "
      "very slow deliberate motion, no abrupt movements, static camera")
B4 = ("She gives the viewer flirty pleading bedroom eyes and slowly mouths a seductive invitation to come "
      "closer and suck, biting her lip sensually, slight slow head tilt, the cock stays erect pointing up, "
      "very slow smooth motion, no abrupt movements, static camera")

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
    "job_id": "ideafix_invite_v2", "character": CHAR, "name": NAME,
    "prompt": "slow seductive talking invitation come closer and suck",
    "clip_prompts": [B1, B2, B3, B4],
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
