from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

from gaze.video import video_duration_s


class VideoAndSubsetTests(unittest.TestCase):
    def test_mp4_mvhd_duration_is_read_without_ffprobe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.mp4"
            path.write_bytes(box(b"ftyp", b"isom0000") + box(b"moov", mvhd(timescale=1000, duration=12345)))
            self.assertAlmostEqual(video_duration_s(path), 12.345)

    def test_aria_projection_is_bounded_and_directional(self) -> None:
        subset = load_subset_script()
        center = subset.aria_yaw_pitch_to_normalized(0.0, 0.0)
        left = subset.aria_yaw_pitch_to_normalized(0.2, 0.0)
        up = subset.aria_yaw_pitch_to_normalized(0.0, 0.2)
        self.assertEqual(center, (0.5, 0.5))
        self.assertLess(left[0], center[0])
        self.assertLess(up[1], center[1])
        for point in [center, left, up, subset.aria_yaw_pitch_to_normalized(10.0, -10.0)]:
            self.assertTrue(0.0 <= point[0] <= 1.0)
            self.assertTrue(0.0 <= point[1] <= 1.0)


def load_subset_script():
    path = Path(__file__).parent.parent / "scripts" / "create_rectify_subset_from_nfs.py"
    spec = importlib.util.spec_from_file_location("create_rectify_subset_from_nfs", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, box_type) + payload


def mvhd(timescale: int, duration: int) -> bytes:
    payload = b"\x00\x00\x00\x00"
    payload += struct.pack(">II", 0, 0)
    payload += struct.pack(">II", timescale, duration)
    payload += b"\x00" * 80
    return box(b"mvhd", payload)


if __name__ == "__main__":
    unittest.main()
