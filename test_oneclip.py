"""One-clip SVI config test. Env: LX2V (lightx2v strength), SVI (svi strength)."""
import os, sys, time
os.environ["RUNPOD_COMFY"] = "https://qa0wdfko120itp-8188.proxy.runpod.net"
import runpod_client
from wan_workflow import build_wan_i2v_workflow

LX2V = float(os.environ.get("LX2V", "1.0"))
SVI = float(os.environ.get("SVI", "1.0"))
STEPS = int(os.environ.get("STEPS", "6"))
TAG = os.environ.get("TAG", f"lx{LX2V}_svi{SVI}_s{STEPS}")

OUTPUT_ROOT = "/home/chremmler/ComfyUI/output/comfy"
png = f"{OUTPUT_ROOT}/cockworship/cw_s1_00002_.png"
prompt = ("She strokes the thick erect cock up and down with her hand in a slow steady rhythm, "
          "continuous stroking motion, the cock stays erect pointing up, languid pace, static camera")

print(f"TEST {TAG}: lightx2v={LX2V} svi={SVI}", flush=True)
comfy_name = runpod_client.upload_image(png)
wf = build_wan_i2v_workflow(
    image_filename=comfy_name, positive_prompt=prompt,
    width=480, height=736, length=81, fps=24, seed=12345,
    long_clip=True, lightx2v_strength=LX2V, svi_strength=SVI, long_steps=STEPS,
    content_loras=[("SmoothFutanaris", 0.7)],
    filename_prefix=f"video/test_{TAG}",
)
t0 = time.time()
pid = runpod_client.post("/prompt", {"prompt": wf})["prompt_id"]
print(f"prompt_id={pid}", flush=True)
videos = runpod_client.poll_until_done(pid)
out = f"/tmp/oneclip_{TAG}.mp4"
runpod_client.download(videos[0]["url"], out)
os.system(f"ffmpeg -y -i {out} -vf 'select=eq(n\\,40)' -vframes 1 /tmp/oneclip_{TAG}.png -loglevel error")
print(f"DONE {TAG} in {time.time()-t0:.0f}s -> /tmp/oneclip_{TAG}.png", flush=True)
