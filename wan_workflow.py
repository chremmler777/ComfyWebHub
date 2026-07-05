"""Builds a ComfyUI API-format WAN 2.2 I2V workflow dict."""
import random

DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走，"
    "shaking penis, vibrating cock, trembling genitals, excessive ball movement, "
    "bouncing balls, jumping testicles, swinging scrotum, bouncing genitals, "
    "hand on penis, touching penis, grabbing penis, stroking penis, hands on genitals, "
    "jittery motion, rapid shaking, uncontrolled vibration, flickering body parts, "
    "sped up motion, fast movement, fast motions, rapid movement, hurried motion, rushed gestures, "
    "teleporting limbs, jerky animation, strobing, abrupt transitions, sudden movements, "
    "camera shake, handheld camera, blurry motion, distorted anatomy, "
    "other people, crowd, background people, multiple people, group, bystanders"
)

# Long-mode negative: hand-on-cock action scenes (stroke, spin). Strips the
# hand-suppression terms and the blanket fast-motion terms so deliberate hand
# action and rotation can happen, while keeping jitter/quality protections.
LONG_NEGATIVE = (
    DEFAULT_NEGATIVE
    .replace("hand on penis, touching penis, grabbing penis, stroking penis, hands on genitals, ", "")
    .replace("sped up motion, fast movement, fast motions, rapid movement, hurried motion, rushed gestures, ", "")
)

# Available content LoRAs — each has a High and Low variant
CONTENT_LORAS = {
    "Anal_Sex":       ("wan/Anal_Sex_High.safetensors",       "wan/Anal_Sex_Low.safetensors"),
    "Breast_Physics": ("wan/Breast_Physics_High.safetensors", "wan/Breast_Physics_Low.safetensors"),
    "PenisLora":      ("wan/PenisLora_High.safetensors",      "wan/PenisLora_Low.safetensors"),
    "Braless":        ("wan/Braless_High.safetensors",         "wan/Braless_Low.safetensors"),
    "BottomTS":       ("wan/Wan_BottomTS_High.safetensors",   "wan/Wan_BottomTS_Low.safetensors"),
    "K3NK_4llinOne":  ("wan/K3NK_4llinOne_High.safetensors",  "wan/K3NK_4llinOne_Low.safetensors"),
    "Bouncing_Boobs": ("wan/Bouncing_Boobs_High.safetensors", "wan/Bouncing_Boobs_Low.safetensors"),
    "WalkToward":     ("wan/WalkToward_High.safetensors",     "wan/WalkToward_Low.safetensors"),
    "BouncyWalk":     ("wan/BouncyWalk_High.safetensors",     "wan/BouncyWalk_Low.safetensors"),
    "DR34ML4Y":       ("wan/DR34ML4Y_I2V_14B_HIGH_V2.safetensors", "wan/DR34ML4Y_I2V_14B_LOW_V2.safetensors"),
    "CumShot":        ("wan/cum4_h_30.safetensors",                "wan/cum4_l_75.safetensors"),
    "SmoothFutanaris": ("wan/SmoothFutanaris_and_Males_High.safetensors", "wan/SmoothFutanaris_and_Males_Low.safetensors"),
    "SlopTwerk":      ("wan/SlopTwerk_High.safetensors",          "wan/SlopTwerk_Low.safetensors"),
}

LIGHTX2V_HIGH = "wan/wan_lightx2v_4steps_high_noise.safetensors"
LIGHTX2V_LOW  = "wan/wan_lightx2v_4steps_low_noise.safetensors"

# SVI 2.0 Pro long-video LoRAs (fp16 rank-128). Auto-loaded only in long mode.
SVI_HIGH = "wan/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors"
SVI_LOW  = "wan/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors"


def build_wan_i2v_workflow(
    image_filename: str,
    positive_prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
    width: int = 640,
    height: int = 640,
    length: int = 81,
    fps: int = 24,
    seed: int | None = None,
    fast_mode: bool = True,
    quality_steps: int = 20,
    content_loras: list[tuple[str, float]] | None = None,
    filename_prefix: str = "video/wan_ai",
    use_rife: bool = False,
    end_image_filename: str | None = None,
    context_windows: bool | None = None,
    context_length: int = 81,
    context_overlap: int = 16,
    long_clip: bool = False,
    lightx2v_strength: float = 0.4,
    svi_strength: float = 1.0,
    long_steps: int = 6,
    prefix_override: str | None = None,
    rife_fps_mult: float = 2.0,  # 2.0=REAL-TIME (162 frames @ 48fps = 3.375s, matches source motion, user-locked). 4/3=opt-in slow-mo money-shot. 1.6 was a slow-mo regression (1.25x slow). Native >81 frames = CGI drift.
) -> dict:
    """
    Returns a ComfyUI API-format workflow dict (for POST /prompt).

    Args:
        image_filename: filename in ComfyUI's input folder (already uploaded)
        positive_prompt: motion/content description
        negative_prompt: what to avoid
        width/height: output resolution (must be multiples of 16)
        length: number of frames (81 ≈ 3.4s at 24fps)
        fps: output video fps
        seed: random seed (None = random)
        fast_mode: True = LightX2V 4-step; False = quality mode
        quality_steps: step count when fast_mode=False (10/20/30)
        content_loras: list of (lora_base_name, strength) e.g. [("Breast_Physics", 0.8)]
        filename_prefix: SaveVideo filename prefix
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    content_loras = content_loras or []
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

    # ── shared: loaders ──────────────────────────────────────
    n_clip  = node("CLIPLoader", {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"})
    n_vae   = node("VAELoader",  {"vae_name": "wan_2.1_vae.safetensors"})
    n_unet_h = node("UNETLoader", {"unet_name": "wan/WAN_High.safetensors", "weight_dtype": "default"})
    n_unet_l = node("UNETLoader", {"unet_name": "wan/WAN_Low.safetensors",  "weight_dtype": "default"})

    # ── shared: prompts ──────────────────────────────────────
    # Prepend motion control prefix to every prompt for smoother, natural movement.
    # Long mode is for hand-on-cock action scenes (stroke, spin), so it drops the
    # "hands away / not touching penis" clause and the heavy slow-motion language.
    if prefix_override is not None:
        # Caller fully controls the prefix (e.g. energetic party clips that must
        # NOT get the slow-motion / hands-away language). Pass "" for no prefix.
        # The negative is used exactly as given (no auto LONG/jerk substitution).
        MOTION_PREFIX = prefix_override
    elif long_clip:
        MOTION_PREFIX = (
            "cinematic, smooth natural motion, static camera, no camera shake, "
            "no teleporting limbs, fluid body motion, "
        )
        if negative_prompt == DEFAULT_NEGATIVE:
            negative_prompt = LONG_NEGATIVE
    elif any(w in positive_prompt.lower() for w in (
            "jerk", "stroke", "strok", "handjob", "hand job", "jacking", "jack off",
            "tugging", "tug", "pumping", "pumps her cock", "pumps his cock",
            "rubbing her cock", "rubbing his cock", "masturbat", "fap", "fisting her cock")):
        # Hand-on-cock action ("jerking"): allow hands on genitals + a rhythmic
        # stroke. Drop the hands-away clause from the prefix and strip the
        # hand-suppression + fast-motion terms from the negative (same as long
        # mode) — but keep the 4-step fast path and the anti-jitter terms so the
        # cock strokes smoothly instead of vibrating.
        MOTION_PREFIX = (
            "cinematic, slow sensual rhythmic stroking, steady smooth rhythm, "
            "static camera, no camera shake, no teleporting limbs, fluid body motion, "
        )
        if negative_prompt == DEFAULT_NEGATIVE:
            negative_prompt = (
                DEFAULT_NEGATIVE
                .replace("hand on penis, touching penis, grabbing penis, stroking penis, hands on genitals, ", "")
                .replace("sped up motion, fast movement, fast motions, rapid movement, hurried motion, rushed gestures, ", "")
            )
    else:
        MOTION_PREFIX = (
            "cinematic, extremely slow and deliberate movement, slow motion, languid pace, "
            "smooth natural motion, static camera, no sudden movements, no fast gestures, "
            "no camera shake, no teleporting limbs, fluid unhurried body motion, "
            "hands away from genitals, not touching penis, "
        )
    # Skip motion prefix for static/frozen shots — it fights against "no movement" intent
    use_prefix = not any(w in positive_prompt.lower() for w in ("frozen", "static portrait", "no movement whatsoever", "completely frozen"))
    n_pos = node("CLIPTextEncode", {"clip": [n_clip, 0], "text": (MOTION_PREFIX if use_prefix else "") + positive_prompt})
    n_neg = node("CLIPTextEncode", {"clip": [n_clip, 0], "text": negative_prompt})

    # ── shared: image ────────────────────────────────────────
    n_img = node("LoadImage", {"image": image_filename})
    n_end_img = node("LoadImage", {"image": end_image_filename}) if end_image_filename else None

    # ── build LoRA chains ────────────────────────────────────
    def build_lora_chain(base_model_id: str, prefix_loras: list[tuple[str, float]],
                         content_loras_for_variant: list[tuple[str, float]]) -> str:
        """Chain: base → [prefix loras (lightx2v, svi)] → [content loras...]."""
        cur = base_model_id
        for lname, strength in prefix_loras + content_loras_for_variant:
            cur = node("LoraLoaderModelOnly", {
                "model": [cur, 0], "lora_name": lname, "strength_model": strength,
            })
        return cur

    content_high, content_low = [], []
    for base_name, strength in content_loras:
        if base_name in CONTENT_LORAS:
            h, l = CONTENT_LORAS[base_name]
            content_high.append((h, strength))
            content_low.append((l, strength))

    prefix_high: list[tuple[str, float]] = []
    prefix_low:  list[tuple[str, float]] = []
    if long_clip:
        prefix_high += [(LIGHTX2V_HIGH, lightx2v_strength), (SVI_HIGH, svi_strength)]
        prefix_low  += [(LIGHTX2V_LOW,  lightx2v_strength), (SVI_LOW,  svi_strength)]
    elif fast_mode:
        prefix_high.append((LIGHTX2V_HIGH, 1.0))
        prefix_low.append((LIGHTX2V_LOW, 1.0))

    n_model_h = build_lora_chain(n_unet_h, prefix_high, content_high)
    n_model_l = build_lora_chain(n_unet_l, prefix_low,  content_low)

    # ── ModelSamplingSD3 ─────────────────────────────────────
    shift = 8.0
    n_samp_h = node("ModelSamplingSD3", {"model": [n_model_h, 0], "shift": shift})
    n_samp_l = node("ModelSamplingSD3", {"model": [n_model_l, 0], "shift": shift})

    # ── Context windows (long video, bounded VRAM) ───────────
    # Auto-enable when length exceeds the model's 81-frame training window.
    # Wraps the model so the sampler processes the long latent in overlapping
    # `context_length`-frame windows and blends them — VRAM stays at one-window
    # cost regardless of total length. standard_static avoids the WAN 2.2
    # uniform-schedule "ping-pong" motion-reversal artifact.
    cw_enabled = False if long_clip else (context_windows if context_windows is not None else (length > 81))
    if cw_enabled:
        cw_args = {
            "context_length":   context_length,
            "context_overlap":  context_overlap,
            "context_schedule": "standard_static",
            "context_stride":   1,
            "closed_loop":      False,
            "fuse_method":      "pyramid",
            "freenoise":        True,
        }
        n_samp_h = node("WanContextWindowsManual", {"model": [n_samp_h, 0], **cw_args})
        n_samp_l = node("WanContextWindowsManual", {"model": [n_samp_l, 0], **cw_args})

    # ── WanImageToVideo ──────────────────────────────────────
    i2v_inputs = {
        "positive":    [n_pos, 0],
        "negative":    [n_neg, 0],
        "vae":         [n_vae, 0],
        "start_image": [n_img, 0],
        "width":  width,
        "height": height,
        "length": length,
        "batch_size": 1,
    }
    if n_end_img:
        i2v_inputs["end_image"] = [n_end_img, 0]
    n_i2v = node("WanImageToVideo", i2v_inputs)

    # ── KSamplers (two-pass: high noise then low noise) ──────
    if long_clip:
        total_steps = long_steps
        mid = long_steps // 2
        n_ks1 = node("KSamplerAdvanced", {
            "model": [n_samp_h, 0], "positive": [n_i2v, 0], "negative": [n_i2v, 1],
            "latent_image": [n_i2v, 2], "add_noise": "enable", "noise_seed": seed,
            "steps": long_steps, "cfg": 1.5, "sampler_name": "euler", "scheduler": "beta",
            "start_at_step": 0, "end_at_step": mid, "return_with_leftover_noise": "enable",
        })
        n_ks2 = node("KSamplerAdvanced", {
            "model": [n_samp_l, 0], "positive": [n_i2v, 0], "negative": [n_i2v, 1],
            "latent_image": [n_ks1, 0], "add_noise": "disable", "noise_seed": seed,
            "steps": long_steps, "cfg": 1.5, "sampler_name": "euler", "scheduler": "beta",
            "start_at_step": mid, "end_at_step": long_steps, "return_with_leftover_noise": "disable",
        })
    elif fast_mode:
        # 4 steps — matches LightX2V LoRA design (LoRA named "4steps"); confirmed better than 6
        # beta scheduler + shift 8.0 reduces flickering vs simple/shift 5.0
        total_steps = 4
        n_ks1 = node("KSamplerAdvanced", {
            "model":         [n_samp_h, 0],
            "positive":      [n_i2v, 0],
            "negative":      [n_i2v, 1],
            "latent_image":  [n_i2v, 2],
            "add_noise":     "enable",
            "noise_seed":    seed,
            "steps":         total_steps,
            "cfg":           1.0,
            "sampler_name":  "euler",
            "scheduler":     "beta",
            "start_at_step": 0,
            "end_at_step":   2,
            "return_with_leftover_noise": "enable",
        })
        n_ks2 = node("KSamplerAdvanced", {
            "model":         [n_samp_l, 0],
            "positive":      [n_i2v, 0],
            "negative":      [n_i2v, 1],
            "latent_image":  [n_ks1, 0],
            "add_noise":     "disable",
            "noise_seed":    seed,
            "steps":         total_steps,
            "cfg":           1.0,
            "sampler_name":  "euler",
            "scheduler":     "beta",
            "start_at_step": 2,
            "end_at_step":   total_steps,
            "return_with_leftover_noise": "disable",
        })
    else:
        # Quality mode — configurable step count, 50/50 split
        total_steps = quality_steps
        mid = total_steps // 2
        n_ks1 = node("KSamplerAdvanced", {
            "model":         [n_samp_h, 0],
            "positive":      [n_i2v, 0],
            "negative":      [n_i2v, 1],
            "latent_image":  [n_i2v, 2],
            "add_noise":     "enable",
            "noise_seed":    seed,
            "steps":         total_steps,
            "cfg":           3.5,
            "sampler_name":  "euler",
            "scheduler":     "beta",
            "start_at_step": 0,
            "end_at_step":   mid,
            "return_with_leftover_noise": "enable",
        })
        n_ks2 = node("KSamplerAdvanced", {
            "model":         [n_samp_l, 0],
            "positive":      [n_i2v, 0],
            "negative":      [n_i2v, 1],
            "latent_image":  [n_ks1, 0],
            "add_noise":     "disable",
            "noise_seed":    seed,
            "steps":         total_steps,
            "cfg":           3.5,
            "sampler_name":  "euler",
            "scheduler":     "beta",
            "start_at_step": mid,
            "end_at_step":   total_steps,
            "return_with_leftover_noise": "disable",
        })

    # ── VAEDecode → ColorMatch → (RIFE) → CreateVideo → SaveVideo ─────────
    n_dec = node("VAEDecode", {"samples": [n_ks2, 0], "vae": [n_vae, 0]})

    # Lock exposure/color to the SOURCE image. WAN fast-mode (LightX2V 4-step)
    # creeps brighter across the clip → blown highlights by the end (esp. high-key
    # scenes like a white outfit). ColorMatch re-anchors every decoded frame's
    # color statistics to the start image, killing the drift. mkl matches mean +
    # covariance; lower `strength` if it over-corrects intentional lighting.
    # ImageColorMatch+ (comfyui_essentials) — self-contained LAB match, no external
    # color-matcher package needed (KJNodes' ColorMatch requires one that isn't
    # installed on the pod). factor 1.0 = full lock to source exposure/color.
    n_dec = node("ImageColorMatch+", {
        "image":       [n_dec, 0],
        "reference":   [n_img, 0],
        "color_space": "LAB",
        "factor":      1.0,
        "device":      "auto",
        "batch_size":  0,
    })

    if use_rife:
        # RIFE doubles the frame count (real frames + interpolated in-betweens).
        # Native length stays 81 (one context window = always photoreal, no
        # "second-window interpretation" drift). Playback default rife_fps_mult=2.0
        # → 162 frames @ 2x fps = REAL-TIME smooth motion (~3.4s at fps=24, 48fps
        # silky), NOT slow-mo. Smoothing only, no diffusion cost. Set rife_fps_mult
        # to 4/3 for a ~5s gentle slow-mo "money shot" extend. (Slow-mo as the
        # default made every clip 1.5x slow — user disliked it, reverted 2026-06-12.)
        n_frames = node("RIFEInterpolation", {
            "images":      [n_dec, 0],
            "source_fps":  float(fps),
            "target_fps":  float(fps * 2),  # 2x frames between real frames
            "scale":       1.0,
            "model_name":  "flownet.pkl",
        })
        n_vid = node("CreateVideo", {"images": [n_frames, 0], "fps": float(fps) * rife_fps_mult})
    else:
        n_vid = node("CreateVideo", {"images": [n_dec, 0], "fps": float(fps)})

    node("SaveVideo", {
        "video":           [n_vid, 0],
        "filename_prefix": filename_prefix,
        "format":          "auto",
        "codec":           "auto",
    })

    return nodes
