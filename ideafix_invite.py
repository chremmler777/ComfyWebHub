"""ideafix talk -> stroke -> point/beckon invite, 3-clip chain."""
import os, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"
import svi_long
from PIL import Image

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
VIDEO_DIR = "/home/chremmler/ComfyUI/output/videos"
CHAR, NAME = "ideas", "ideafix_65425222_00005_"

B1 = ("She talks seductively while looking directly at the viewer, her lips moving as she speaks, "
      "slight head movement, seductive expression, one hand resting on the thick erect cock, "
      "the cock stays erect pointing up, static camera")
B2 = ("She slowly begins stroking the thick erect cock up and down with her hand while looking at the "
      "viewer with a seductive gaze, continuous slow stroking motion, the cock stays erect pointing up, "
      "static camera")
B3 = ("She raises her hand and points her finger at the viewer then curls it in a beckoning come-closer "
      "gesture inviting the viewer in, seductive smile, the cock stays erect pointing up, static camera")

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
    "job_id": "ideafix_invite_1", "character": CHAR, "name": NAME,
    "prompt": "talk stroke beckon viewer closer",
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
