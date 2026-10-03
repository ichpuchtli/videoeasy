"""Inventory records from synthetic ffprobe JSON: creation_time kept as written, camera make/model where the file says."""
import unittest
from pathlib import Path

from videoeasy import probe


def info(fmt_tags=None, video_tags=None, audio=True):
    streams = [{"codec_type": "video", "r_frame_rate": "24000/1001", "width": 3840, "height": 2160, "codec_name": "hevc",
                "tags": video_tags or {}}]
    if audio:
        streams.append({"codec_type": "audio", "tags": {}})
    return {"format": {"duration": "12.5", "tags": fmt_tags or {}}, "streams": streams}


class Probe(unittest.TestCase):
    def test_proapps_tags(self):
        r = probe.record(Path("/x/CAMA0001.MOV"), "broll", info(
            {"creation_time": "2026-03-13T22:39:17.000000Z", "com.apple.proapps.manufacturer": "MakerCo",
             "com.apple.proapps.modelname": "Model One", "comment": "MAKERCO DIGITAL CAMERA Model One"},
            {"creation_time": "2026-03-13T22:39:17.000000Z", "timecode": "13:47:47:21"}))
        self.assertEqual(r["creation_time"], "2026-03-13T22:39:17.000000Z")
        self.assertEqual((r["camera_make"], r["camera_model"]), ("MakerCo", "Model One"))
        self.assertEqual(r["start_timecode"], "13:47:47:21")
        self.assertAlmostEqual(r["fps"], 23.976, places=3)

    def test_comment_only_camera(self):
        r = probe.record(Path("/x/CAMB0001.MOV"), "aroll", info({"creation_time": "2026-03-13T23:35:27Z",
                                                               "comment": "MAKERCO DIGITAL CAMERA Model Two"}))
        self.assertEqual((r["camera_make"], r["camera_model"]), ("MAKERCO", "Model Two"))

    def test_drone_encoder_and_remuxed_file(self):
        r = probe.record(Path("/x/DJI_1001.MP4"), "broll", info({"creation_time": "2026-03-10T22:33:02Z", "encoder": "DJI Air"}))
        self.assertEqual((r["camera_make"], r["camera_model"]), ("DJI", "Air"))
        r = probe.record(Path("/x/DJI_1002.MOV"), "broll", info({"creation_time": "2026-03-10T22:33:02Z", "encoder": "Lavf56.15.102"}))
        self.assertEqual((r["camera_make"], r["camera_model"]), (None, None))

    def test_creation_time_from_a_stream_when_the_container_has_none(self):
        r = probe.record(Path("/x/C1.MOV"), "broll", info({}, {"Creation_Time": "2026-03-13T01:00:00Z"}))
        self.assertEqual(r["creation_time"], "2026-03-13T01:00:00Z")
        r = probe.record(Path("/x/C2.MOV"), "broll", info({}, {}, audio=False))
        self.assertEqual((r["creation_time"], r["camera_make"], r["has_audio"]), (None, None, False))


if __name__ == "__main__":
    unittest.main()
