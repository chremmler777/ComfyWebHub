"""megan singing performance, 2-clip chain (variant out_stem)."""
import os, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"
import svi_long
from PIL import Image

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
VIDEO_DIR = "/home/chremmler/ComfyUI/output/videos"
CHAR, NAME = "megan", "megan_plane_stroke_open_00006_"

B1 = ("She sings passionately, her mouth opening and moving as she sings, head swaying and bobbing to "
      "the music, expressive emotive singing face, one hand raised holding a microphone near her mouth, "
      "the other hand gesturing with the song, smooth performing motion, the erect cock stays rigid and "
      "still keeping its solid shape, static camera, no fast jerky movements")
B2 = ("She continues her passionate singing performance, mouth moving with the lyrics, head and "
      "shoulders swaying to the rhythm, eyes closing then opening toward the viewer, holding the "
      "microphone, expressive emotive face, the erect cock stays rigid and still, smooth performing "
      "motion, static camera, no fast jerky movements")

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
    "job_id": "megan_sing_1", "character": CHAR, "name": NAME,
    "out_stem": "megan_plane_stroke_open_00006_singing",
    "prompt": "singing performance on stage",
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
