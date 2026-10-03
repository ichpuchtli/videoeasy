"""`videoeasy doctor` reporting with a fake machine: no network, no subprocess, no Resolve."""
import unittest
from pathlib import Path

from videoeasy import doctor
from videoeasy.vlm import MIN_CONTEXT_TOKENS


class FakeEnv(doctor.Env):
    def __init__(self, **kw):
        self.plat = kw.get("plat", ("darwin", "arm64"))
        self.tools = kw.get("tools", {"ffmpeg", "ffprobe", "uv", "lms"})
        self.lms_loaded = kw.get("lms_loaded", ["google/gemma-4-26b-a4b"])
        self.lms_up = kw.get("lms_up", True)
        self.ctx = kw.get("ctx", 16384)
        self.ollama = kw.get("ollama", ["qwen3.8:27b-mtp-q8_0"])
        self.specs = kw.get("specs", {"mlx_whisper"})
        self.paths = kw.get("paths", "all")
        self.resolve = kw.get("resolve", True)
        self.free = kw.get("free", 500e9)

    def platform(self):
        return self.plat

    def python(self):
        return (3, 12, 1)

    def which(self, tool):
        return f"/usr/bin/{tool}" if tool in self.tools else None

    def first_line(self, argv):
        return f"{argv[0]} version 1" if argv[0] in self.tools else None

    def get_json(self, url):
        if url.endswith("/api/v0/models"):
            return {"data": [{"id": m, "state": "loaded"} for m in self.lms_loaded]} if self.lms_up else None
        if url.endswith("/api/tags"):
            return None if self.ollama is None else {"models": [{"name": m} for m in self.ollama]}
        return None

    def context_length(self, url, model):
        return self.ctx

    def find_spec(self, name):
        return name in self.specs

    def exists(self, path):
        return self.paths == "all"

    def disk_free(self, path):
        return self.free

    def resolve_answers(self):
        return self.resolve


def run(env, models=("qwen3.8:27b-mtp-q8_0",)):
    return doctor.report(doctor.checks(env, ollama_models=models, repo_root=Path("/repo")))


class Doctor(unittest.TestCase):
    def test_all_present(self):
        text, code = run(FakeEnv())
        self.assertEqual(code, 0)
        self.assertNotIn("MISSING", text)
        self.assertIn("all required items present", text)

    def test_missing_required_fails_with_a_fix(self):
        text, code = run(FakeEnv(tools={"ffprobe", "uv", "lms"}, specs=set()))
        self.assertEqual(code, 1)
        self.assertIn("MISSING  ffmpeg", text)
        self.assertIn("fix: brew install ffmpeg", text)
        self.assertIn("fix: uv sync", text)

    def test_small_context_window_is_missing_not_a_warning(self):
        text, code = run(FakeEnv(ctx=4096))
        self.assertEqual(code, 1)
        self.assertIn("4096-token window", text)
        self.assertIn(f"--context-length {MIN_CONTEXT_TOKENS}", text)

    def test_vision_model_not_loaded_and_server_down(self):
        self.assertIn("is not loaded", run(FakeEnv(lms_loaded=[]))[0])
        text, code = run(FakeEnv(lms_up=False))
        self.assertEqual(code, 1)
        self.assertIn("lms server start", text)

    def test_optional_items_only_warn(self):
        text, code = run(FakeEnv(ollama=None, resolve=False, free=20e9))
        self.assertEqual(code, 0)
        self.assertIn("WARN     Ollama", text)
        self.assertIn("WARN     DaVinci Resolve", text)
        self.assertIn("WARN     free disk", text)
        text, code = run(FakeEnv(ollama=[]))
        self.assertEqual(code, 0)
        self.assertIn("ollama pull qwen3.8:27b-mtp-q8_0", text)

    def test_not_apple_silicon(self):
        text, code = run(FakeEnv(plat=("linux", "x86_64")))
        self.assertEqual(code, 1)
        self.assertIn("mlx-whisper", text)

    def test_resolve_not_installed_is_a_warning(self):
        text, code = run(FakeEnv(paths="none"))
        self.assertIn("scripting module not found", text)
        self.assertIn("whisper weights", text)


if __name__ == "__main__":
    unittest.main()
