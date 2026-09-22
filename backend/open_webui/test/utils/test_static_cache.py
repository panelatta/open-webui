import pytest
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.testclient import TestClient

from open_webui.main import SPAStaticFiles


@pytest.fixture
def static_site(tmp_path):
    files = {
        'index.html': '<html>Current app</html>',
        'manifest.json': '{}',
        '_app/version.json': '{"version":"current"}',
        '_app/immutable/entry/start.Abc12345.js': 'export const version = 1;',
        '_app/immutable/assets/app.Abc12345.css': 'body { color: black; }',
        '_app/immutable/workers/worker-Abc12345.js': 'self.close();',
    }
    for name, content in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    app = Starlette(routes=[Mount('/', SPAStaticFiles(directory=tmp_path, html=True))])
    with TestClient(app) as client:
        yield client, tmp_path


@pytest.mark.parametrize('method', ['get', 'head'])
@pytest.mark.parametrize('path', [
    '/_app/immutable/entry/start.Abc12345.js',
    '/_app/immutable/assets/app.Abc12345.css',
    '/_app/immutable/workers/worker-Abc12345.js',
])
def test_generated_assets_allow_long_term_cache(static_site, method, path):
    client, _ = static_site
    response = getattr(client, method)(path)
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'public, max-age=31536000, immutable'
    assert 'etag' in response.headers
    assert 'pragma' not in response.headers


@pytest.mark.parametrize('path', [
    '/', '/index.html', '/manifest.json', '/_app/version.json',
    '/c/example-chat', '/_app/immutable/assets/missing.Abc12345.css',
])
def test_mutable_files_and_spa_fallback_are_never_cached(static_site, path):
    client, _ = static_site
    response = client.get(path)
    assert response.status_code == 200
    assert 'no-store' in response.headers['cache-control']
    assert 'etag' not in response.headers
    assert 'last-modified' not in response.headers


def test_missing_javascript_is_not_cached_as_success(static_site):
    client, _ = static_site
    response = client.get('/_app/immutable/chunks/missing.Abc12345.js')
    assert response.status_code == 404
    assert 'immutable' not in response.headers.get('cache-control', '')


def test_error_document_is_not_immutable(static_site):
    client, directory = static_site
    (directory / '404.html').write_text('Not found')
    response = client.get('/_app/immutable/chunks/missing.Abc12345.js')
    assert response.status_code == 404
    assert 'no-store' in response.headers['cache-control']
