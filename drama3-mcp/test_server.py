"""MCP contract checks against a small Studio HTTP stand-in, without models."""

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import httpx
from mcp.shared.memory import create_connected_server_and_client_session

from server import build_server

UPLOAD, CROP, ASR, JOB = (char * 32 for char in "abcd")


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.audio = Path(folder.name) / "test audio.wav"
        self.audio.write_bytes(b"test-audio" * 8)
        self.requests = []
        self.queued = []
        self.transcriptions = []
        self.voice_design = True
        self.words = {UPLOAD: "One last train. The doors are closing.", CROP: "one last train"}

    def studio(self, request):
        path = request.url.path
        payload = json.loads(request.content) if request.headers.get("content-type") == "application/json" else {}
        self.requests.append((path, payload))
        if path == "/api/model":
            value = {"selected_ready": True, "selected_directory": "model.gguf"}
        elif path == "/api/capabilities":
            value = dict.fromkeys(("scene_scheduler", "joint_dialogue", "audio_import", "infill", "director_revision"), True)
            value["voice_design"] = self.voice_design
        elif path == "/api/reference-audio":
            self.assertIn(b'name="audio"; filename="test audio.wav"', request.content)
            value = {"id": UPLOAD}
        elif path == "/api/reference-audio/crop":
            self.assertEqual(payload["id"], UPLOAD)
            value = {"id": CROP}
        elif path == "/api/transcriptions":
            self.assertIs(payload["local_only"], True)
            identifier = payload["upload_id"]
            self.transcriptions.append(identifier)
            value = ({"id": ASR, "status": "complete", "result": {
                "text": self.words[identifier], "language": "English", "duration_seconds": 20 if identifier == UPLOAD else 2.4}}
                     if identifier in self.words else {"id": ASR, "status": "skipped"})
        elif path in ("/api/jobs", "/api/audio-edit/jobs") and request.method == "POST":
            self.queued.append((path, payload))
            value = {"id": JOB}
        elif path == f"/api/jobs/{JOB}":
            value = {"status": "complete", "output": str(self.audio), "result": {"duration_seconds": 2.4}}
        else:
            self.fail(f"Unexpected Studio request: {request.method} {path}")
        return httpx.Response(200, json=value)

    @asynccontextmanager
    async def session(self):
        class Studio:
            @asynccontextmanager
            async def connect(studio):
                async with httpx.AsyncClient(base_url="http://127.0.0.1:8879",
                                             transport=httpx.MockTransport(self.studio)) as client:
                    yield client

        async with create_connected_server_and_client_session(build_server(Studio())) as client:
            yield client

    async def call(self, client, name, **arguments):
        response = await client.call_tool(name, arguments)
        self.assertFalse(response.isError, response.content)
        return response.structuredContent

    async def prepared(self, client, submission):
        self.assertEqual(submission["status"], "preparing")
        for _ in range(100):
            job = await self.call(client, "get_job", job_id=submission["job_id"])
            if job["status"] != "preparing":
                return job
            await asyncio.sleep(0.01)
        self.fail("MCP preparation did not finish")

    async def test_script_guide_is_discoverable_without_studio(self):
        async with self.session() as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            self.assertTrue(tools["get_script_guide"].annotations.readOnlyHint)
            for name in ("submit_speech", "submit_voice_design", "submit_dialogue"):
                self.assertIn("get_script_guide", tools[name].description)
            result = await self.call(client, "get_script_guide")
        for marker in ("Fish Audio S2", "[empathetic]", "<whisper>", "S2"):
            self.assertIn(marker, result["guide"])
        self.assertFalse(self.requests)

    async def test_voice_design_separates_timbre_from_delivery(self):
        async with self.session() as client:
            submitted = await self.call(client, "submit_voice_design", voice_description="A low, grainy voice.",
                                        text="[calm] <soft>Come in.</soft>", direction="Welcome a longtime friend.")
            self.assertEqual(submitted, {"job_id": JOB, "status": "queued"})
            result = await self.call(client, "get_job", job_id=JOB)
        self.assertEqual(result["wav_path"], str(self.audio))
        self.assertEqual(self.queued, [("/api/jobs", {
            "mode": "design", "voice_description": "A low, grainy voice.", "text": "[calm] <soft>Come in.</soft>",
            "direction": "Welcome a longtime friend.", "voice_bindings": [], "model": "model.gguf",
            "automatic_length": True, "silence_retry": True,
        })])
        self.assertFalse(self.transcriptions)

    async def test_voice_design_requires_model_support(self):
        self.voice_design = False
        async with self.session() as client:
            result = await client.call_tool("submit_voice_design", {"voice_description": "A low voice.", "text": "Hi."})
        self.assertTrue(result.isError)
        self.assertIn("does not support voice design", result.content[0].text)
        self.assertFalse(self.queued)

    async def test_standalone_transcription_of_full_file_and_selection(self):
        async with self.session() as client:
            for identifier, duration, bounds in ((UPLOAD, 20, {}), (CROP, 2.4, {"start_seconds": 128.2, "end_seconds": 130.6})):
                result = await self.prepared(client, await self.call(client, "transcribe_audio",
                                             audio_path=str(self.audio), **bounds))
                self.assertEqual(result, {
                    "job_id": result["job_id"], "status": "complete", "text": self.words[identifier],
                    "language": "English", "duration_seconds": duration,
                })
        self.assertEqual(self.transcriptions, [UPLOAD, CROP])
        self.assertIn(("/api/reference-audio/crop", {"id": UPLOAD, "start": 128.2, "end": 130.6}), self.requests)
        self.assertNotIn("/api/model", [path for path, _ in self.requests])
        self.assertFalse(self.queued)
        self.assertEqual(self.audio.read_bytes(), b"test-audio" * 8)

    async def test_standalone_transcription_reports_missing_asr_and_invalid_selection(self):
        self.words.clear()
        async with self.session() as client:
            invalid = await client.call_tool("transcribe_audio", {"audio_path": str(self.audio), "start_seconds": 8.2})
            self.assertTrue(invalid.isError)
            self.assertFalse(self.requests)
            result = await self.prepared(client, await self.call(client, "transcribe_audio", audio_path=str(self.audio)))
        self.assertEqual(result["status"], "needs_input")
        self.assertIn("Install the local ASR model in Studio", result["error"])
        self.assertNotIn("manually", result["error"])
        self.assertFalse(self.queued)

    async def test_dialogue_preserves_role_assignments_and_inline_tags(self):
        async with self.session() as client:
            result = await self.call(client, "submit_dialogue",
                                     script="S2\uff1a[calm] Ready?\nS1: <soft>Yes.</soft>",
                                     voices={"S1": "one", "S2": "two"}, strategy="per_line")
            missing = await client.call_tool("submit_dialogue", {"script": "S1: Ready?\nS2: Yes.", "voices": {"S1": "one"}})
        self.assertTrue(missing.isError)
        self.assertEqual(result, {"job_id": JOB, "status": "queued"})
        self.assertEqual(len(self.queued), 1)
        path, payload = self.queued[0]
        self.assertEqual((path, payload["mode"], payload["scene_strategy"]), ("/api/jobs", "scene", "per_line"))
        self.assertEqual(payload["lines"], [{"character": "S2", "text": "[calm] Ready?"},
                                            {"character": "S1", "text": "<soft>Yes.</soft>"}])
        self.assertEqual(payload["voice_bindings"], [{"voice_id": "two", "speaker": 1}, {"voice_id": "one", "speaker": 0}])

    async def test_local_clone_transcribes_or_uses_supplied_words(self):
        async with self.session() as client:
            for words in (None, "Verified reference words."):
                job = await self.prepared(client, await self.call(client, "submit_speech", text="New words.",
                                          reference_audio={"audio_path": str(self.audio), "transcript": words}))
                self.assertEqual(job["wav_path"], str(self.audio))
        self.assertEqual(self.transcriptions, [UPLOAD])
        self.assertEqual([payload["voice_bindings"][0]["text"] for _, payload in self.queued],
                         [self.words[UPLOAD], "Verified reference words."])

    async def test_missing_asr_does_not_submit_unconditioned_speech(self):
        self.words.clear()
        async with self.session() as client:
            job = await self.prepared(client, await self.call(client, "submit_speech", text="New words.",
                                      reference_audio={"audio_path": str(self.audio)}))
        self.assertEqual(job["status"], "needs_input")
        self.assertIn("reference transcript", job["error"])
        self.assertEqual(self.queued, [])

    async def test_timed_edit_uses_full_revised_transcript_and_original_audio(self):
        async with self.session() as client:
            job = await self.prepared(client, await self.call(client, "submit_audio_edit", audio_path=str(self.audio),
                                      start_seconds=8.2, end_seconds=10.6, text="The next train."))
        self.assertEqual(job["status"], "complete")
        self.assertEqual(self.transcriptions, [UPLOAD, CROP])
        path, payload = self.queued[0]
        self.assertEqual((path, payload["operation"], payload["upload_id"]), ("/api/audio-edit/jobs", "infill", UPLOAD))
        self.assertEqual((payload["start_frame"], payload["end_frame"]), (176, 229))
        self.assertEqual(payload["text"], "The next train. The doors are closing.")

    async def test_repeated_excerpt_requires_confirmation_before_edit(self):
        self.words[UPLOAD] = "One last train. One last train."
        async with self.session() as client:
            args = {"audio_path": str(self.audio), "start_seconds": 8.2, "end_seconds": 10.6, "text": "The next train."}
            job = await self.prepared(client, await self.call(client, "submit_audio_edit", **args))
            self.assertEqual(job["status"], "needs_input")
            self.assertFalse(self.queued)
            job = await self.prepared(client, await self.call(client, "submit_audio_edit", **args,
                                      original_transcript=self.words[UPLOAD], original_excerpt="One last train", occurrence=2))
        self.assertEqual(job["status"], "complete")
        self.assertEqual(self.queued[0][1]["text"], "One last train. The next train.")
