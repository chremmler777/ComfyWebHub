# Keeper Web — WAN I2V Video Pipeline

## What This Is

Two-part system for AI video generation from images:

1. **Local keeperweb** (`/home/chremmler/claude/comfy2/keeper_web/`) — Flask UI at `http://127.0.0.1:5151`
   - Browse ComfyUI output images by character
   - Mark keepers, star, add notes, trigger HQ passes
   - Animate images → WAN 2.2 I2V videos via RunPod

2. **RunPod ComfyUI** — GPU pod at ~$0.49/hr
   - WAN 2.2 I2V model (High + Low noise two-pass)
   - Bootstrap repo: `https://github.com/chremmler777/comfy-bootstrap`

## How to Start

### 1. Start local keeperweb

```bash
cd /home/chremmler/claude/comfy2/keeper_web
python app.py
```

Open `http://127.0.0.1:5151` — browse images, `http://127.0.0.1:5151/videos` — local videos.

### 2. Create RunPod pod (when needed)

```bash
cd /home/chremmler/claude/runpod
RUNPOD_API_KEY=xxx CIVITAI_TOKEN=xxx python create_pod.py --gpu A4000
```

- Creates pod with 80GB volume (required — 50GB fills up fast)
- Pod prints SSH + proxy URLs on creation
- Wait ~10-15 min for model downloads on first run

### 3. Connect keeperweb to RunPod

Set `RUNPOD_COMFY` env var to the pod's 8188 proxy URL before starting:

```bash
RUNPOD_COMFY="https://{pod_id}-8188.proxy.runpod.net" \
RUNPOD_POD_ID="{pod_id}" \
RUNPOD_API_KEY=xxx \
python app.py
```

Or set it in the running process — currently hardcoded to last pod as default.

### 4. Check RunPod is ready

```bash
curl "https://{pod_id}-8188.proxy.runpod.net/queue"
```

Should return `{"queue_running": [], "queue_pending": []}`.

## Animate Workflow (How Jobs Flow)

```
User clicks Animate on image
    → keeperweb creates "pending" job in memory (/api/animate)
    → Claude reads image + user description in chat
    → Claude crafts refined WAN I2V prompt
    → Claude calls /api/jobs/dispatch/{job_id} with refined prompt
    → keeperweb uploads image to RunPod, submits to ComfyUI
    → status: queued → running → done
    → keeperweb auto-downloads video to VIDEO_DIR
    → video appears at /videos page
```

### Dispatching a pending job (Claude does this)

```bash
curl -s http://127.0.0.1:5151/api/jobs | python3 -c "import json,sys; [print(j['job_id'][:8], j['status'], j['character'], j['name']) for j in json.load(sys.stdin)]"
```

Then dispatch with a crafted prompt:
```bash
curl -X POST http://127.0.0.1:5151/api/jobs/dispatch/{job_id} \
  -H "Content-Type: application/json" \
  -d '{"prompt": "YOUR REFINED WAN PROMPT HERE"}'
```

### Rerun a video with feedback

In `/videos` page — click **Rerun**, add feedback, submit.
Creates a new pending job combining original prompt + feedback.
Claude dispatches it with an improved prompt.

## WAN I2V Prompting Rules

- Image defines appearance — **only describe motion, never appearance**
- 80-120 words, specific action verbs with pace adverbs
- Explicit camera: `"static shot"` prevents drift; omit camera for handheld feel
- Subtle > dramatic — prevents identity/morphing artifacts
- Include secondary motion: hair, fabric, environment
- Negative prompt always includes: `morphing, warping, flickering, face deformation`

**Example good prompt:**
> Static shot. Woman slowly inhales, chest rising as she draws breath, stomach tightening subtly. Her fingers rest motionless on her thighs while her eyes blink naturally once. Hair shifts barely from breath movement. Subtle chest heave rhythm continues. Fabric of clothing catches micro-tension with each breath. Camera locked off completely. Negative: morphing, warping, flickering, identity drift, face deformation, limb distortion.

**Content LoRAs available** (use in workflow only when relevant):
- `Breast_Physics` — breast physics/jiggle
- `Braless` — braless appearance + physics
- `PenisLora` — male anatomy
- `Anal_Sex` — anal sex motion
- `K3NK_4llinOne` — general intimate motion
- `BottomTS` — butt motion

All LoRA names need `wan/` prefix in workflow: `wan/Breast_Physics.safetensors`

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Flask app, port 5151, all endpoints |
| `wan_workflow.py` | Builds ComfyUI API workflow dict for WAN I2V |
| `planner.py` | Claude API prompt planner (optional, not used in manual flow) |
| `static/index.html` | Image browser + animate dialog |
| `static/videos.html` | Local video grid + rerun dialog |
| `static/queue.html` | Job queue monitor |

## Env Vars

| Var | Default | Notes |
|-----|---------|-------|
| `RUNPOD_COMFY` | last pod URL | RunPod ComfyUI proxy URL |
| `RUNPOD_POD_ID` | (empty) | Pod ID for auto-shutdown |
| `RUNPOD_API_KEY` | (empty) | RunPod key for auto-shutdown |
| `ANTHROPIC_API_KEY` | (empty) | Only needed for `/api/plan_prompt` |

## Auto-Shutdown

When `RUNPOD_POD_ID` and `RUNPOD_API_KEY` are set, keeperweb runs a background thread that:
- Checks every 60 seconds
- When **all jobs are done/error** AND **ComfyUI queue is empty**
- Calls RunPod `podTerminate` GraphQL mutation
- Prints `[auto-shutdown] All jobs done, terminating pod {id}`

## Disk Space

RunPod volume **must be 80GB** — 50GB fills up quickly with models + outputs.
- WAN High model: ~9GB
- WAN Low model: ~9GB
- LoRAs: ~200MB each
- If disk fills: `pkill -f 'python.*main.py'` on RunPod to free deleted file handles

## Output Locations

- **Source images**: `/home/chremmler/ComfyUI/output/comfy/{character}/`
- **Downloaded videos**: `/home/chremmler/ComfyUI/output/videos/{character}/`
- **Video metadata sidecars**: same dir, `.json` files with prompt/settings
- **Video thumbnails**: `keeper_web/data/video_thumbs/`

## RunPod Bootstrap

Bootstrap installs on pod via `install.sh`:
```bash
# On RunPod pod:
curl -fsSL https://raw.githubusercontent.com/chremmler777/comfy-bootstrap/main/install.sh | bash
```

Or SSH in and run manually. Script:
1. Clones bootstrap repo
2. Downloads WAN I2V models (High + Low fp8)
3. Installs ComfyUI + custom nodes
4. Deploys keeperweb (RunPod version, port 8189)
5. Starts both services

## Troubleshooting

**Videos not showing up**: Check `/api/jobs` — if status stuck at `queued`, poll status manually:
```bash
curl http://127.0.0.1:5151/api/animate/status/{prompt_id}
```

**Job list empty after restart**: Jobs are in-memory only. Re-dispatch or check RunPod history directly at `{RUNPOD_COMFY}/history`.

**ComfyUI 400 error**: Usually invalid workflow node inputs. `control_after_generate` is NOT valid for `KSamplerAdvanced`.

**Model not found**: All model names need `wan/` prefix: `wan/WAN_High.safetensors`, `wan/Breast_Physics.safetensors`.

**Fullscreen video collapses**: Fixed — videos are downloaded locally and served with `conditional=True` for Range request support.
