"""SVI 2.0 Pro long-video orchestrator: chain N 81-frame clips into one MP4."""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time

import runpod_client
from wan_workflow import build_wan_i2v_workflow

CLIP_FRAMES = 81
MAX_CLIPS = 7
MIN_CLIPS = 4
SVI_SHORT_SIDE = 480

# Words that imply an evolving / multi-beat scene (more clips).
_BUILDUP_WORDS = ("build", "then", "speeds up", "faster", "finally", "cums", "climax", "starts", "transition")


def pick_clip_count(prompt: str) -> int:
    """Heuristic clip count from prompt content. Loops → MIN, evolving → MAX. Capped at MAX_CLIPS."""
    p = (prompt or "").lower()
    if any(w in p for w in _BUILDUP_WORDS):
        return MAX_CLIPS
    return MIN_CLIPS


def svi_dims(width: int, height: int) -> tuple[int, int]:
    """Scale so the short side is SVI_SHORT_SIDE, preserve aspect, round to multiples of 16."""
    def r16(x: float) -> int:
        return max(16, int(round(x / 16)) * 16)
    if width <= height:
        return SVI_SHORT_SIDE, r16(SVI_SHORT_SIDE * height / width)
    return r16(SVI_SHORT_SIDE * width / height), SVI_SHORT_SIDE


def _last_frame_cmd(clip_mp4: str, out_png: str) -> list[str]:
    """ffmpeg command to write the last frame of a clip as a single PNG."""
    return ["ffmpeg", "-y", "-sseof", "-0.1", "-i", clip_mp4,
            "-update", "1", "-frames:v", "1", out_png]


def _concat_cmd(clip_paths: list[str], out_mp4: str) -> list[str]:
    """ffmpeg filter_complex: keep clip 0 whole; trim 1 leading frame off clips 1..n
    (that frame is the seed == previous clip's last frame); concat all."""
    cmd = ["ffmpeg", "-y"]
    for p in clip_paths:
        cmd += ["-i", p]
    parts, labels = [], []
    for i in range(len(clip_paths)):
        lbl = f"v{i}"
        if i == 0:
            parts.append(f"[{i}:v]setpts=PTS-STARTPTS[{lbl}]")
        else:
            parts.append(f"[{i}:v]trim=start_frame=1,setpts=PTS-STARTPTS[{lbl}]")
        labels.append(f"[{lbl}]")
    filt = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(clip_paths)}:v=1:a=0[out]"
    cmd += ["-filter_complex", filt, "-map", "[out]", out_mp4]
    return cmd


def extract_last_frame(clip_mp4: str, out_png: str) -> str:
    subprocess.run(_last_frame_cmd(clip_mp4, out_png), capture_output=True, timeout=60, check=True)
    return out_png


def concat_clips(clip_paths: list[str], out_mp4: str) -> str:
    subprocess.run(_concat_cmd(clip_paths, out_mp4), capture_output=True, timeout=300, check=True)
    return out_mp4


_THUMBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "video_thumbs")


def _publish_to_keeperweb(final_mp4: str, char: str, stem: str, job: dict) -> None:
    """Generate the thumbnail + json sidecar keeperweb needs so the finished video shows up."""
    os.makedirs(_THUMBS_DIR, exist_ok=True)
    thumb = os.path.join(_THUMBS_DIR, f"{char}__{stem}.jpg")
    subprocess.run(["ffmpeg", "-y", "-i", final_mp4, "-vframes", "1", "-q:v", "3", thumb],
                   capture_output=True, timeout=30)
    sidecar = final_mp4[:-4] + ".json"
    with open(sidecar, "w") as f:
        json.dump({"character": char, "src_name": job.get("name", ""),
                   "prompt": job.get("prompt", ""), "fps": int(job.get("fps", 24)),
                   "long_mode": True, "clips": job.get("clip_progress", "")}, f)


def run_long_job(job: dict, *, output_root, video_dir, ensure_loras, on_update) -> None:
    """Render the chained clips and concat into one MP4.

    Uses job['clip_prompts'] (list, one per clip) when present, else job['prompt']
    for every clip. `ensure_loras(content_loras)` injects always-on LoRAs
    (SmoothFutanaris). `on_update()` persists job state. Runs in a background thread.
    """
    try:
        clip_prompts = job.get("clip_prompts") or []
        if clip_prompts:
            n = len(clip_prompts)
        else:
            n = int(job.get("target_clips") or pick_clip_count(job.get("prompt", "")))
        n = max(1, min(MAX_CLIPS, n))

        w, h = svi_dims(int(job["width"]), int(job["height"]))
        base_seed = int(time.time()) & 0xFFFFFFFF
        char, name = job.get("character", "unknown"), job.get("name", "clip")
        start_png = os.path.join(str(output_root), char, f"{name}.png")

        tmp = tempfile.mkdtemp(prefix="svi_")
        clips = []
        job["status"] = "running"
        on_update()
        for i in range(n):
            prompt = clip_prompts[i] if clip_prompts else job["prompt"]
            comfy_name = runpod_client.upload_image(start_png)
            wf = build_wan_i2v_workflow(
                image_filename=comfy_name, positive_prompt=prompt,
                width=w, height=h, length=CLIP_FRAMES, fps=int(job.get("fps", 24)),
                seed=base_seed + i, long_clip=True,
                # SVI off by default: its loras corrupt generation via the standard
                # LoraLoaderModelOnly path (needs Kijai native WanVideoWrapper nodes).
                # Chaining alone (last-frame seeding) carries continuity. Override per job.
                lightx2v_strength=float(job.get("lightx2v_strength", 1.0)),
                svi_strength=float(job.get("svi_strength", 0.0)),
                long_steps=int(job.get("long_steps", 6)),
                content_loras=ensure_loras(job.get("content_loras", [])),
                filename_prefix=f"video/svi_{job['job_id']}_{i}",
            )
            pid = runpod_client.post("/prompt", {"prompt": wf})["prompt_id"]
            videos = runpod_client.poll_until_done(pid)
            clip_path = os.path.join(tmp, f"clip_{i}.mp4")
            runpod_client.download(videos[0]["url"], clip_path)
            clips.append(clip_path)
            if i < n - 1:
                start_png = extract_last_frame(clip_path, os.path.join(tmp, f"seed_{i+1}.png"))
            job["clip_progress"] = f"{i+1}/{n}"
            on_update()

        dest_dir = os.path.join(str(video_dir), char)
        os.makedirs(dest_dir, exist_ok=True)
        # out_stem lets multiple long videos from the same source image coexist
        # (e.g. a "dance" variant vs an "invite" variant) instead of overwriting.
        stem = f"{job.get('out_stem') or name}__svi_long"
        final = os.path.join(dest_dir, f"{stem}.mp4")
        concat_clips(clips, final)
        _publish_to_keeperweb(final, char, stem, job)
        shutil.rmtree(tmp, ignore_errors=True)  # drop temp clip parts
        job["status"] = "done"
        job["videos"] = [{"rel": f"{char}/{stem}.mp4", "url": ""}]
        on_update()
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        on_update()


def start(job: dict, **kwargs) -> None:
    threading.Thread(target=run_long_job, args=(job,), kwargs=kwargs, daemon=True).start()
