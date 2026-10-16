"""Aria RGB gaze is 90° CW from VRS/calib onto frame-aligned mp4s."""

from egoexo4d.media import rotate_gaze_cw90, scale_gaze


def test_rotate_gaze_cw90_square_center():
    (x, y), wh = rotate_gaze_cw90((5.0, 5.0), (11.0, 11.0))
    assert (x, y) == (5.0, 5.0)
    assert wh == (11.0, 11.0)


def test_rotate_gaze_cw90_bottom_to_left():
    (x, y), wh = rotate_gaze_cw90((5.0, 10.0), (11.0, 11.0))
    assert (x, y) == (0.0, 5.0)
    assert wh == (11.0, 11.0)


def test_scale_after_rotate_matches_video():
    xy, cal_wh = rotate_gaze_cw90((704.0, 1407.0), (1408.0, 1408.0))
    gx, gy = scale_gaze(xy, cal_wh, (1408, 1408))
    assert abs(gx - 0.0) < 1e-6
    assert abs(gy - 704.0) < 1e-6
