import tempfile
import unittest
from pathlib import Path

from PIL import Image

from videoeasy.frames import contact_sheet


class ContactSheetTests(unittest.TestCase):
    def test_labels_do_not_cover_or_reorder_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, color in enumerate(("red", "blue")):
                path = root / f"f{index:02}.png"
                Image.new("RGB", (80, 40), color).save(path)
                paths.append(path)
            output = root / "sheet.png"
            contact_sheet(paths, output, cols=2, tile_w=80, labels=["f00", "f01"])
            with Image.open(output) as sheet:
                self.assertEqual(sheet.size, (160, 70))
                self.assertEqual(sheet.getpixel((40, 20)), (255, 0, 0))
                self.assertEqual(sheet.getpixel((120, 20)), (0, 0, 255))
                self.assertIsNotNone(sheet.crop((0, 40, 80, 70)).getbbox())
                self.assertGreater(len(sheet.crop((0, 40, 80, 70)).getcolors()), 1)

    def test_rejects_missing_label(self):
        with self.assertRaisesRegex(ValueError, "one label"):
            contact_sheet([Path("unused")], Path("unused"), labels=[])

    def test_rejects_empty_frames(self):
        with self.assertRaises(ValueError):
            contact_sheet([], Path("unused"))


if __name__ == "__main__":
    unittest.main()
