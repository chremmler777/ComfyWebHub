import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import svi_long


def test_pick_clip_count_loop_is_small():
    assert svi_long.pick_clip_count("she strokes in a continuous rhythmic loop") == 4


def test_pick_clip_count_buildup_is_large():
    n = svi_long.pick_clip_count("slow build up then she speeds up and finally cums")
    assert n == 7


def test_pick_clip_count_default():
    assert svi_long.pick_clip_count("standing still, gentle breathing") == 4


def test_pick_clip_count_capped():
    assert svi_long.pick_clip_count("x " * 500) <= 7


def test_svi_dims_portrait_480_short_side():
    w, h = svi_long.svi_dims(640, 640)
    assert w == 480 and h == 480


def test_svi_dims_preserves_aspect_and_mult16():
    w, h = svi_long.svi_dims(1080, 1920)  # portrait
    assert w == 480
    assert h % 16 == 0
    assert abs((h / w) - (1920 / 1080)) < 0.05


def test_extract_last_frame_cmd():
    cmd = svi_long._last_frame_cmd("clip.mp4", "frame.png")
    assert cmd[0] == "ffmpeg"
    assert "clip.mp4" in cmd and "frame.png" in cmd
    assert "-update" in cmd  # single-image output


def test_concat_drops_first_frame_after_first_clip():
    cmd = svi_long._concat_cmd(["a.mp4", "b.mp4", "c.mp4"], "out.mp4")
    s = " ".join(cmd)
    assert cmd[0] == "ffmpeg"
    assert cmd[-1] == "out.mp4"
    assert "a.mp4" in s and "b.mp4" in s and "c.mp4" in s
    assert s.count("trim=start_frame=1") == 2
    assert "concat=n=3" in s


def test_concat_single_clip_no_trim():
    cmd = svi_long._concat_cmd(["only.mp4"], "out.mp4")
    s = " ".join(cmd)
    assert "trim=start_frame=1" not in s
    assert "only.mp4" in s
