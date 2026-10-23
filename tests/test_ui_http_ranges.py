"""Real HTTP seek ranges/HEAD, including invalid and suffix requests."""

from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import ui.server as server


@pytest.fixture
def audio_server(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    audio = tmp_path / "outputs" / "example.wav"
    audio.parent.mkdir()
    audio.write_bytes(bytes(range(256)) * 16)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/files/outputs/example.wav", audio.read_bytes()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("range_value,start,end", [
    ("bytes=0-99", 0, 99), ("bytes=1024-2047", 1024, 2047),
    ("bytes=4080-", 4080, 4095), ("bytes=-16", 4080, 4095),
    ("bytes=4000-9000", 4000, 4095),
])
def test_byte_ranges_serve_exact_audio_bytes(audio_server, range_value, start, end):
    url, expected = audio_server
    with urlopen(Request(url, headers={"Range": range_value})) as response:
        assert response.status == 206
        assert response.headers["Content-Range"] == f"bytes {start}-{end}/{len(expected)}"
        assert response.headers["Accept-Ranges"] == "bytes"
        assert int(response.headers["Content-Length"]) == end - start + 1
        assert response.read() == expected[start:end + 1]


@pytest.mark.parametrize("range_value", ["bytes=9000-", "bytes=100-1", "bytes=-0",
                                       "bytes=0-1,3-4", "bytes=-", "items=1-2"])
def test_bad_audio_ranges_are_rejected_without_sending_a_whole_file(audio_server, range_value):
    url, expected = audio_server
    with pytest.raises(HTTPError) as result:
        urlopen(Request(url, headers={"Range": range_value}))
    response = result.value
    assert response.code == 416
    assert response.headers["Content-Range"] == f"bytes */{len(expected)}"
    assert response.read() == b""


def test_head_exposes_audio_size_and_seek_support_without_transferring_pcm(audio_server):
    url, expected = audio_server
    with urlopen(Request(url, method="HEAD")) as response:
        assert response.status == 200 and response.read() == b""
        assert int(response.headers["Content-Length"]) == len(expected)
        assert response.headers["Accept-Ranges"] == "bytes"
    with urlopen(url) as response:
        assert response.status == 200 and response.read() == expected
