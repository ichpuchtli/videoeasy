import unittest
from unittest import mock

import httpx

from videoeasy import vlm


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class LoadedContextLength(unittest.TestCase):
    def test_reads_lm_studio_model_listing(self):
        listing = {"data": [{"id": "other", "loaded_context_length": 99},
                            {"id": "google/gemma-4-26b-a4b", "loaded_context_length": 4096}]}
        with mock.patch.object(httpx, "get", return_value=FakeResponse(listing)) as get:
            self.assertEqual(vlm.loaded_context_length("http://localhost:1234/v1", "google/gemma-4-26b-a4b"), 4096)
        self.assertEqual(get.call_args.args[0], "http://localhost:1234/api/v0/models")

    def test_unknown_when_server_lacks_endpoint(self):
        with mock.patch.object(httpx, "get", side_effect=httpx.ConnectError("down")):
            self.assertIsNone(vlm.loaded_context_length("http://localhost:1234/v1", "m"))
        with mock.patch.object(httpx, "get", return_value=FakeResponse({"data": []})):
            self.assertIsNone(vlm.loaded_context_length("http://localhost:1234/v1", "m"))

    def test_minimum_leaves_room_for_eight_frames(self):
        self.assertGreaterEqual(vlm.MIN_CONTEXT_TOKENS, 8192)


if __name__ == "__main__":
    unittest.main()
