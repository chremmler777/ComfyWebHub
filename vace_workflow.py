"""WAN 2.1 VACE workflow builder — reference-anchored video + temporal extension.

VACE is a SINGLE-model architecture (not the 2.2 high/low-noise split), so this is
separate from wan_workflow.py. The key feature: `reference_image` anchors the
subject's identity in EVERY generated frame, which fixes the identity drift that
plain context-window i2v showed after the first window.

Speed: uses the WAN 2.1 lightx2v step-distill LoRA → 4-step generation.

Models on the volume:
  diffusion_models/wan/wan2.1_vace_14B_fp16.safetensors
  loras/wan/Wan21_T2V_14B_lightx2v_distill_rank32.safetensors
  vae/wan_2.1_vae.safetensors  |  text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
"""
import random

VACE_UNET   = "wan/wan2.1_vace_14B_fp16.safetensors"
VACE_SPEED_LORA = "wan/Wan21_T2V_14B_lightx2v_distill_rank32.safetensors"

DEFAULT_NEGATIVE_VACE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "morphing, warping, distortion, flickering, deformed, blurry, low quality, "
    "extra limbs, shaking, jitter, vibrating genitals"
)


def build_vace_workflow(
    reference_image: str,
    positive_prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_VACE,
    width: int = 480,
    height: int = 832,
    length: int = 81,
    fps: int = 24,
    seed: int | None = None,
    steps: int = 4,
    cfg: float = 1.0,
    filename_prefix: str = "video/vace_ai",
    keep_video: str | None = None,   # uploaded mp4 whose frames are the leading "keep" frames (overlap)
    n_keep: int = 1,                  # number of leading frames to preserve (1 = anchor to ref for chunk0)
    speed_lora: bool = True,
) -> dict:
    """Build a ComfyUI API workflow for one VACE chunk.

    chunk 0:  keep_video=None, n_keep=1  -> frame 0 anchored to reference_image, rest generated.
    chunk N:  keep_video=<overlap.mp4>, n_keep=<frames in it> -> continues from those frames.
    reference_image anchors identity across the whole clip in every case.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    nodes: dict = {}
    _nid = [1]

    def nid():
        n = str(_nid[0]); _nid[0] += 1; return n

    def node(class_type, inputs):
        n = nid(); nodes[n] = {"class_type": class_type, "inputs": inputs}; return n

    # ── loaders ──────────────────────────────────────────────
    n_clip = node("CLIPLoader", {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"})
    n_vae  = node("VAELoader",  {"vae_name": "wan_2.1_vae.safetensors"})
    n_unet = node("UNETLoader", {"unet_name": VACE_UNET, "weight_dtype": "default"})

    # speed LoRA (4-step distill)
    n_model = n_unet
    if speed_lora:
        n_model = node("LoraLoaderModelOnly", {"model": [n_model, 0], "lora_name": VACE_SPEED_LORA, "strength_model": 1.0})
    n_model = node("ModelSamplingSD3", {"model": [n_model, 0], "shift": 8.0})

    # ── prompts ──────────────────────────────────────────────
    n_pos = node("CLIPTextEncode", {"clip": [n_clip, 0], "text": positive_prompt})
    n_neg = node("CLIPTextEncode", {"clip": [n_clip, 0], "text": negative_prompt})

    # ── reference image (identity anchor) ────────────────────
    n_ref = node("LoadImage", {"image": reference_image})

    # ── build control_video (length frames) + control_masks ──
    #   keep frames (mask 0, preserved)  ++  generate frames (mask 1, grey)
    n_gen = length - n_keep
    # keep images
    if keep_video:
        n_keepimgs = node("VHS_LoadVideo", {
            "video": keep_video, "force_rate": 0, "force_size": "Disabled",
            "frame_load_cap": n_keep, "skip_first_frames": 0, "select_every_nth": 1,
        })
        keep_src = [n_keepimgs, 0]
    else:
        keep_src = [n_ref, 0]  # chunk 0: single reference frame anchors frame 0

    # grey filler for the to-generate region
    n_grey = node("EmptyImage", {"width": width, "height": height, "batch_size": n_gen, "color": 8421504})  # mid grey
    n_ctrlvid = node("ImageBatch", {"image1": keep_src, "image2": [n_grey, 0]})

    # masks: build as images then convert — black(keep=0) ++ white(generate=1)
    n_black = node("EmptyImage", {"width": width, "height": height, "batch_size": n_keep, "color": 0})
    n_white = node("EmptyImage", {"width": width, "height": height, "batch_size": n_gen, "color": 16777215})
    n_maskimg = node("ImageBatch", {"image1": [n_black, 0], "image2": [n_white, 0]})
    n_mask = node("ImageToMask", {"image": [n_maskimg, 0], "channel": "red"})

    # ── VACE encode ──────────────────────────────────────────
    n_vace = node("WanVaceToVideo", {
        "positive": [n_pos, 0],
        "negative": [n_neg, 0],
        "vae": [n_vae, 0],
        "width": width, "height": height, "length": length, "batch_size": 1,
        "strength": 1.0,
        "control_video": [n_ctrlvid, 0],
        "control_masks": [n_mask, 0],
        "reference_image": [n_ref, 0],
    })

    # ── sample (single model, 4-step distill) ────────────────
    n_ks = node("KSampler", {
        "model": [n_model, 0],
        "positive": [n_vace, 0],
        "negative": [n_vace, 1],
        "latent_image": [n_vace, 2],
        "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": "euler", "scheduler": "beta", "denoise": 1.0,
    })

    # ── trim the reference frames VACE prepended, decode, save ─
    n_trim = node("TrimVideoLatent", {"samples": [n_ks, 0], "trim_amount": [n_vace, 3]})
    n_dec = node("VAEDecode", {"samples": [n_trim, 0], "vae": [n_vae, 0]})
    n_vid = node("CreateVideo", {"images": [n_dec, 0], "fps": float(fps)})
    node("SaveVideo", {"video": [n_vid, 0], "filename_prefix": filename_prefix, "format": "auto", "codec": "auto"})

    return nodes
