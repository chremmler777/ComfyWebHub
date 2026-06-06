import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wan_workflow import build_wan_i2v_workflow, SVI_HIGH, SVI_LOW

def _classes(wf):
    return [n["class_type"] for n in wf.values()]

def _lora_names(wf):
    return [n["inputs"]["lora_name"] for n in wf.values() if n["class_type"] == "LoraLoaderModelOnly"]

def test_long_clip_uses_six_steps_cfg_1_5():
    wf = build_wan_i2v_workflow("img.png", "a futa strokes", long_clip=True, seed=1)
    ksamplers = [n for n in wf.values() if n["class_type"] == "KSamplerAdvanced"]
    assert len(ksamplers) == 2
    assert all(k["inputs"]["steps"] == 6 for k in ksamplers)
    assert all(k["inputs"]["cfg"] == 1.5 for k in ksamplers)
    assert ksamplers[0]["inputs"]["end_at_step"] == 3
    assert ksamplers[1]["inputs"]["start_at_step"] == 3

def test_long_clip_injects_svi_loras():
    wf = build_wan_i2v_workflow("img.png", "x", long_clip=True, svi_strength=1.0, seed=1)
    names = _lora_names(wf)
    assert SVI_HIGH in names and SVI_LOW in names

def test_long_clip_reduces_lightx2v():
    wf = build_wan_i2v_workflow("img.png", "x", long_clip=True, lightx2v_strength=0.4, seed=1)
    lx = [n for n in wf.values()
          if n["class_type"] == "LoraLoaderModelOnly" and "lightx2v" in n["inputs"]["lora_name"]]
    assert lx and all(n["inputs"]["strength_model"] == 0.4 for n in lx)

def test_long_clip_disables_context_windows():
    wf = build_wan_i2v_workflow("img.png", "x", long_clip=True, length=81, seed=1)
    assert "WanContextWindowsManual" not in _classes(wf)

def test_short_path_unchanged_four_steps():
    wf = build_wan_i2v_workflow("img.png", "x", fast_mode=True, seed=1)
    ks = [n for n in wf.values() if n["class_type"] == "KSamplerAdvanced"]
    assert all(k["inputs"]["steps"] == 4 and k["inputs"]["cfg"] == 1.0 for k in ks)
    assert SVI_HIGH not in _lora_names(wf)
