"""satsuki cute Japanese idol-style dance, 2-clip chain (variant out_stem)."""
import os, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"
import svi_long
from PIL import Image

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
VIDEO_DIR = "/home/chremmler/ComfyUI/output/videos"
CHAR, NAME = "satsuki", "satsuki_conv_lift_skirt_00001_"

B1 = ("She does a cute playful Japanese idol-style dance, swaying her hips side to side and moving her "
      "arms in cute rhythmic J-pop dance gestures, bouncing lightly to a beat, her white angel wings "
      "flutter and her long black hair sways, bright playful smile, the erect cock stays rigid and still "
      "with no jiggle, smooth rhythmic dancing motion, static camera")
B2 = ("She continues the cute Japanese idol dance, doing playful arm poses and a little hip shimmy, "
      "bouncing rhythmically to the beat, wings fluttering, hair swaying, happy playful smile, the erect "
      "cock stays rigid and still, smooth rhythmic dancing motion, static camera")

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
    "job_id": "satsuki_dance_1", "character": CHAR, "name": NAME,
    "out_stem": "satsuki_conv_lift_skirt_00001_dance",
    "prompt": "cute japanese idol style dance",
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
