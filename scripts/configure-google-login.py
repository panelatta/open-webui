"""Configure Google OAuth for the local bind-mount deployment, without logging secrets."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def configure(directory, credentials, origin):
    url = urlsplit(origin)
    if url.scheme != 'https' or not url.netloc or url.path not in ('', '/') or url.query or url.fragment or url.username:
        raise ValueError('Use the public HTTPS origin without a path')
    origin = 'https://' + url.netloc
    web = json.loads(credentials.read_text()).get('web', {})
    values = {
        'WEBUI_URL': origin,
        'ENABLE_OAUTH': 'true',
        'ENABLE_OAUTH_SIGNUP': 'true',
        'ENABLE_OAUTH_PERSISTENT_CONFIG': 'false',
        'GOOGLE_CLIENT_ID': web.get('client_id', ''),
        'GOOGLE_CLIENT_SECRET': web.get('client_secret', ''),
        'GOOGLE_REDIRECT_URI': origin + '/oauth/google/callback',
        'OAUTH_AUTO_REDIRECT': 'false',
        'OAUTH_MERGE_ACCOUNTS_BY_EMAIL': 'false',
        'OAUTH_UPDATE_EMAIL_ON_LOGIN': 'false',
    }
    if not values['GOOGLE_CLIENT_ID'] or not values['GOOGLE_CLIENT_SECRET']:
        raise ValueError('A Web application client ID and secret are required')
    if any(any(c in v for c in ["'", "\n", "\r"]) for v in values.values()):
        raise ValueError('Unexpected characters in OAuth configuration')
    compose = directory / 'docker-compose.yml'
    content = compose.read_text()
    entry = '    env_file:\n      - ./google-oauth.env\n'
    if entry not in content:
        if '    env_file:' in content:
            raise ValueError('Review the existing env_file block before adding Google OAuth')
        anchor = '    environment:\n'
        if content.count(anchor) != 1:
            raise ValueError('Expected one open-webui environment block')
        content = content.replace(anchor, entry + anchor, 1)
    output = directory / 'google-oauth.env'
    fd = os.open(output, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(''.join(k + "='" + v + "'\n" for k,v in values.items()))
    os.chmod(output, 0o600)
    os.chmod(credentials, 0o600)
    compose.write_text(content)
    print('Configured Google OAuth environment; credentials were not printed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment-dir', type=Path, required=True)
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--origin', required=True)
    args = parser.parse_args()
    configure(args.deployment_dir, args.credentials, args.origin)
