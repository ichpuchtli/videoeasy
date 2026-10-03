"""Event items as film directories: the project-level film.yaml fallback, source paths that do not assume a layout,
and the text-model override."""
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from videoeasy import cutbuild, layin, textmodel
from videoeasy.film import film_dir_for, load_film


class ProjectProfile(unittest.TestCase):
    def setUp(self):
        self.project = Path(tempfile.mkdtemp()).resolve()
        (self.project / "film.yaml").write_text('name: "Event"\nlayin: {project: "Event project", render_dir: editorial/renders}\n')
        self.item = self.project / "items" / "feature-01"
        (self.item / "editorial").mkdir(parents=True)

    def test_item_without_its_own_profile_reads_the_projects(self):
        f = load_film(self.item)
        self.assertEqual(f.name, "Event")
        self.assertEqual(f.source_path, self.project / "film.yaml")
        self.assertEqual(f.root, self.item)
        self.assertEqual(f.layin["render_dir"], str(self.item / "editorial/renders"))   # renders stay per item

    def test_items_own_profile_wins(self):
        (self.item / "film.yaml").write_text('name: "Feature one"\n')
        f = load_film(self.item)
        self.assertEqual((f.name, f.source_path), ("Feature one", self.item / "film.yaml"))

    def test_no_profile_anywhere_is_generic_and_says_so(self):
        lone = Path(tempfile.mkdtemp())
        self.assertIsNone(load_film(lone).source_path)
        self.assertIsNone(load_film(None).source_path)

    def test_film_dir_for_finds_the_item_not_the_project(self):
        cut = self.item / "editorial" / "cut-v1.json"
        cut.write_text("{}")
        self.assertEqual(film_dir_for(cut), self.item)
        self.assertEqual(film_dir_for(self.project / "deliverables.yaml"), self.project)


class SourcePaths(unittest.TestCase):
    def test_cutbuild_uses_the_recorded_path(self):
        film = Path("/films/f")
        self.assertEqual(cutbuild.unit_source_path(film, {"source": "DJI_1001", "role": "broll", "source_path": "/cards/DJI_1001.MP4"}),
                         "/cards/DJI_1001.MP4")
        self.assertEqual(cutbuild.unit_source_path(film, {"source": "CAMA0001", "role": "broll"}),
                         "/films/f/inputs/broll/CAMA0001.MOV")

    def test_layin_falls_back_to_the_inventory(self):
        root = Path(tempfile.mkdtemp())
        (root / "out").mkdir()
        (root / "out/inventory.json").write_text(json.dumps({"aroll": [{"id": "CAMA0001", "path": "/cards/CAMA0001.MP4"}], "broll": []}))
        inv = layin.inventory_paths(root)
        cfg = {"aroll_dir": "/films/f/inputs/aroll"}
        self.assertEqual(layin.pick_source_path({}, "CAMA0001", cfg, inv), "/cards/CAMA0001.MP4")
        self.assertEqual(layin.pick_source_path({"source_path": "/given.MOV"}, "CAMA0001", cfg, inv), "/given.MOV")
        self.assertEqual(layin.pick_source_path({}, "CAMA0002", cfg, inv), "/films/f/inputs/aroll/CAMA0002.MOV")
        self.assertEqual(layin.inventory_paths(None), {})


class TextModel(unittest.TestCase):
    def test_default_and_override(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VIDEOEASY_TEXT_MODEL", None)
            os.environ.pop("VIDEOEASY_TEXT_URL", None)
            self.assertEqual(textmodel.text_model(), textmodel.DEFAULT_TEXT_MODEL)
            self.assertEqual(textmodel.text_url(), textmodel.DEFAULT_TEXT_URL)
        with mock.patch.dict(os.environ, {"VIDEOEASY_TEXT_MODEL": "other:7b", "VIDEOEASY_TEXT_URL": "http://box:11434/"}):
            self.assertEqual(textmodel.text_model(), "other:7b")
            self.assertEqual(textmodel.text_url(), "http://box:11434")

    def test_tools_resolve_it_at_start(self):
        from videoeasy import storycheck
        try:
            with mock.patch.dict(os.environ, {"VIDEOEASY_TEXT_MODEL": "other:7b"}):
                importlib.reload(storycheck)
                self.assertEqual(storycheck.OLLAMA_MODEL, "other:7b")
        finally:
            importlib.reload(storycheck)
        self.assertEqual(storycheck.OLLAMA_MODEL, textmodel.text_model())


if __name__ == "__main__":
    unittest.main()
