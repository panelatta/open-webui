import os
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
OPEN_WEBUI_DIR = BACKEND_DIR / 'open_webui'

for path in (BACKEND_DIR, OPEN_WEBUI_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault('GOOGLE_CLOUD_PROJECT', 'open-webui-test')
os.environ.setdefault('STORAGE_EMULATOR_HOST', 'http://127.0.0.1:9023')
os.environ.setdefault('AZURE_STORAGE_ENDPOINT', 'http://127.0.0.1:10000/open-webui-test')
os.environ.setdefault('AZURE_STORAGE_CONTAINER_NAME', 'open-webui-test')
os.environ.setdefault('AZURE_STORAGE_KEY', 'open-webui-test-key')
if not os.environ.get('WEBUI_SECRET_KEY'):
    os.environ['WEBUI_SECRET_KEY'] = 'open-webui-test-secret'
