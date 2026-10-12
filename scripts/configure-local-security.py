"""Prepare the existing bind-mount deployment without reading secrets into logs."""
import argparse
import os
from pathlib import Path
from urllib.parse import urlsplit


def configure(directory, origin, build_hash, port, proxy_gateway):
    parsed = urlsplit(origin)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.path not in ('', '/'):
        raise ValueError('The public origin must be an HTTPS origin without a path')
    origin = f'{parsed.scheme}://{parsed.netloc}'
    updates = {
        'OPEN_WEBUI_PORT': f'127.0.0.1:{port}',
        'WEBUI_BUILD_HASH': build_hash,
        'STATIC_DIR': '/app/backend/data/static',
        'CORS_ALLOW_ORIGIN': f'{origin};http://localhost:{port};http://127.0.0.1:{port}',
        'WEBUI_AUTH_COOKIE_SECURE': 'true',
        'WEBUI_SESSION_COOKIE_SECURE': 'true',
        'ENABLE_PASSWORD_VALIDATION': 'true',
        'PASSWORD_VALIDATION_REGEX_PATTERN': r'^[\s\S]{12,}$',
        'PASSWORD_VALIDATION_HINT': 'Use at least 12 characters.',
        'FORWARDED_ALLOW_IPS': f'127.0.0.1,{proxy_gateway}',
    }
    env = directory / '.env'
    lines = env.read_text().splitlines()
    seen = set()
    for index, line in enumerate(lines):
        key = line.split('=', 1)[0]
        if key in updates:
            lines[index] = key + '=' + updates[key]
            seen.add(key)
    lines.extend(key+'='+value for key, value in updates.items() if key not in seen)
    compose = directory / 'docker-compose.yml'
    text = compose.read_text()
    anchor = '    environment:\n'
    if anchor not in text:
        raise ValueError('Expected the existing open-webui Compose environment block')
    additions = ''.join('      '+key+': ${'+key+'}\n' for key in updates
                        if key != 'OPEN_WEBUI_PORT' and '\n      '+key+':' not in text)
    text = text.replace(anchor, anchor+additions, 1)
    source_mount = '../open-webui-src/backend/open_webui:/app/backend/open_webui'
    if source_mount+':ro' not in text:
        if source_mount+'\n' not in text:
            raise ValueError('Expected the existing backend bind mount')
        text = text.replace(source_mount+'\n', source_mount+':ro\n', 1)
    # Validate everything before writing either file. Back up both files before calling.
    env.write_text('\n'.join(lines)+'\n')
    os.chmod(env, 0o600)
    compose.write_text(text)
    print('Configured loopback port, exact CORS origins, secure cookies, password policy and read-only source mount.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment-dir', type=Path, required=True)
    parser.add_argument('--origin', required=True)
    parser.add_argument('--build-hash', required=True)
    parser.add_argument('--port', type=int, default=3000)
    parser.add_argument('--proxy-gateway', required=True)
    args = parser.parse_args()
    configure(args.deployment_dir, args.origin, args.build_hash, args.port, args.proxy_gateway)
