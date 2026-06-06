"""Builds a ComfyUI API-format LTXV 2.3 Distilled I2V workflow dict."""
import random

CHECKPOINT    = "lightricksLTXV23_distilled.safetensors"
TEXT_ENCODER  = "comfy_gemma_3_12B_it.safetensors"
DISTILLED_LORA = "ltxv/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
# 9-step sigma schedule for the distilled model (from official example workflow)
DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

DEFAULT_NEGATIVE = (
    "low quality, worst quality, deformed, distorted, disfigured, motion smear, "
    "motion artifacts, fused fingers, bad anatomy, weird hand, ugly, blurry, "
    "watermark, subtitle, text"
)


def build_ltxv_i2v_workflow(
    image_filename: str,
    positive_prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
    width: int = 768,
    height: int = 512,
    length: int = 97,   # (length-1) must be divisible by 8; 97 ≈ 4s @ 24fps
    fps: int = 24,
    seed: int | None = None,
    lora_strength_high: float = 0.5,
    lora_strength_low: float = 0.2,
    image_strength: float = 0.85,
    cfg: float = 1.0,
    filename_prefix: str = "video/ltxv_ai",
) -> dict:
    """
    Returns a ComfyUI API-format workflow dict (for POST /prompt).

    Args:
        image_filename:    filename in ComfyUI's input folder (already uploaded)
        positive_prompt:   motion/content description
        negative_prompt:   what to avoid
        width/height:      output resolution — must be multiples of 32
        length:            frame count; (length-1) must be divisible by 8
                           valid values: 9,17,25,33,41,49,57,65,73,81,89,97,105,113,121
        fps:               output fps
        seed:              random seed (None = random)
        lora_strength_high: first LoRA pass strength (default 0.5)
        lora_strength_low:  second LoRA pass strength (default 0.2)
        image_strength:    how strongly the start image anchors the generation
        cfg:               classifier-free guidance (1.0 for distilled)
        filename_prefix:   SaveVideo filename prefix
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    nodes: dict = {}
    _nid = [1]

    def nid() -> str:
        n = str(_nid[0])
        _nid[0] += 1
        return n

    def node(class_type: str, inputs: dict) -> str:
        n = nid()
        nodes[n] = {"class_type": class_type, "inputs": inputs}
        return n

    # ── Model + VAE ──────────────────────────────────────────
    n_ckpt = node("CheckpointLoaderSimple", {"ckpt_name": CHECKPOINT})
    # outputs: [model=0, clip=1, vae=2]

    # ── Text encoder (Gemma 3 12B) ───────────────────────────
    n_te = node("LTXAVTextEncoderLoader", {
        "text_encoder": TEXT_ENCODER,
        "ckpt_name":    CHECKPOINT,
        "device":       "default",
    })
    # outputs: [clip=0]

    # ── LoRA chain (two passes of the distilled LoRA) ────────
    n_lora1 = node("LoraLoaderModelOnly", {
        "model":         [n_ckpt, 0],
        "lora_name":     DISTILLED_LORA,
        "strength_model": lora_strength_high,
    })
    n_lora2 = node("LoraLoaderModelOnly", {
        "model":         [n_lora1, 0],
        "lora_name":     DISTILLED_LORA,
        "strength_model": lora_strength_low,
    })

    # ── Text conditioning ────────────────────────────────────
    n_pos = node("CLIPTextEncode", {"text": positive_prompt, "clip": [n_te, 0]})
    n_neg = node("CLIPTextEncode", {"text": negative_prompt, "clip": [n_te, 0]})

    # LTXVConditioning wraps text conditioning with fps info
    n_cond = node("LTXVConditioning", {
        "positive":   [n_pos, 0],
        "negative":   [n_neg, 0],
        "frame_rate": float(fps),
    })
    # outputs: [positive=0, negative=1]

    # ── Image + latent ───────────────────────────────────────
    n_img    = node("LoadImage", {"image": image_filename})
    n_latent = node("EmptyLTXVLatentVideo", {
        "width":      width,
        "height":     height,
        "length":     length,
        "batch_size": 1,
    })

    # Encode start image into the latent
    n_i2v = node("LTXVImgToVideoConditionOnly", {
        "vae":      [n_ckpt, 2],
        "image":    [n_img, 0],
        "latent":   [n_latent, 0],
        "strength": image_strength,
        "bypass":   False,
    })
    # outputs: [latent=0]

    # ── Sampler (distilled path) ─────────────────────────────
    n_guider  = node("CFGGuider", {
        "model":    [n_lora2, 0],
        "positive": [n_cond, 0],
        "negative": [n_cond, 1],
        "cfg":      cfg,
    })
    n_sampler = node("KSamplerSelect", {"sampler_name": "euler_ancestral_cfg_pp"})
    n_sigmas  = node("ManualSigmas",   {"sigmas": DISTILLED_SIGMAS})
    n_noise   = node("RandomNoise",    {"noise_seed": seed})

    n_result = node("SamplerCustomAdvanced", {
        "noise":        [n_noise, 0],
        "guider":       [n_guider, 0],
        "sampler":      [n_sampler, 0],
        "sigmas":       [n_sigmas, 0],
        "latent_image": [n_i2v, 0],
    })
    # outputs: [output=0, denoised_output=1]

    # ── Decode + save ────────────────────────────────────────
    n_dec = node("LTXVTiledVAEDecode", {
        "vae":              [n_ckpt, 2],
        "latents":          [n_result, 0],
        "horizontal_tiles": 2,
        "vertical_tiles":   2,
        "overlap":          6,
        "last_frame_fix":   False,
    })
    # outputs: [image=0]

    n_vid = node("CreateVideo", {"images": [n_dec, 0], "fps": float(fps)})
    node("SaveVideo", {
        "video":           [n_vid, 0],
        "filename_prefix": filename_prefix,
        "format":          "auto",
        "codec":           "auto",
    })

    return nodes
