from types import SimpleNamespace

import pytest

from open_webui.utils.session_pool import stream_wrapper


@pytest.mark.anyio
async def test_upstream_stream_wrapper_preserves_split_sse_lines_and_closes_response():
    long_line = b'data: ' + b'x' * 100000 + b'\n'
    chunks = [b'data: {"type":"response.', b'created"}\n\n', long_line[:50000], long_line[50000:], b'data: [DONE]']

    async def iter_chunks():
        for chunk in chunks:
            yield chunk, False

    response = SimpleNamespace(content=SimpleNamespace(iter_chunks=iter_chunks), closed=False)

    def close():
        response.closed = True

    response.close = close
    lines = [line async for line in stream_wrapper(response)]
    assert lines == [b'data: {"type":"response.created"}\n', b'\n', long_line, b'data: [DONE]']
    assert response.closed
