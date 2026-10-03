"""Repair the existing GLM preset through the model API, preserving other settings.

Run inside the Open WebUI container with --backup PATH; add --apply to save.
The backup is created exclusively and contains the original model API response.
No database writes, local search, or embedding calls are made by this script.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import sqlite3
import time

import jwt
import requests


TOOL_GUIDANCE = """工具使用与交付要求：
- 用户要求联网调查时，实际调用可用的 web_search 工具，根据搜索结果回答并提供来源链接。
- create_tasks/update_task 只用于记录当前会话的任务进度，不会启动后台执行器。创建任务清单后，继续执行可用工具并在本轮给出结果；完成的任务应更新状态。
- 不要把“现在开始检索”等开场白或计划作为最终回答，也不要声称未执行的搜索已经完成。
- 如果工具不可用或调用失败，明确说明限制或错误，并交付能够确认的部分；不要承诺会在结束回答后继续后台执行。"""


def configure(model):
    model = copy.deepcopy(model)
    params = model.setdefault('params', {})
    custom = params.setdefault('custom_params', {})
    tools = custom.get('tools', [])
    if isinstance(tools, str):
        tools = json.loads(tools)
    if not isinstance(tools, list):
        raise ValueError('Existing custom tools must be a list')
    if not any(t.get('type') in {'web_search', 'web_search_preview'} for t in tools):
        tools.append({'type': 'web_search'})
    custom['tools'] = tools
    custom.setdefault('tool_choice', 'auto')
    system = params.get('system') or ''
    if TOOL_GUIDANCE not in system:
        params['system'] = '\n\n'.join(filter(None, [system, TOOL_GUIDANCE]))
    # This capability describes the model. The global local-search switch and
    # builtinTools.web_search are deliberately left alone: search is hosted.
    model.setdefault('meta', {}).setdefault('capabilities', {})['web_search'] = True
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', default='ablit-glm53-custom')
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    data_dir = Path(os.environ.get('DATA_DIR', '/app/backend/data'))
    with sqlite3.connect(f'file:{data_dir}/webui.db?mode=ro', uri=True) as db:
        user = db.execute("select id from user where role='admin' order by created_at limit 1").fetchone()
    if not user:
        raise RuntimeError('No admin account found')
    session = requests.Session()
    session.headers['Authorization'] = 'Bearer ' + jwt.encode(
        {'id': user[0], 'exp': int(time.time()) + 300}, os.environ['WEBUI_SECRET_KEY'], algorithm='HS256'
    )
    url = 'http://127.0.0.1:8080/api/v1/models/model'
    response = session.get(url, params={'id': args.model_id}, timeout=30)
    response.raise_for_status()
    original = response.json()
    configured = configure(original)
    with args.backup.open('x') as backup:
        os.chmod(args.backup, 0o600)
        json.dump(original, backup, ensure_ascii=False, indent=2)
    if args.apply:
        response = session.post(url + '/update', params={'id': args.model_id}, json=configured, timeout=30)
        response.raise_for_status()
        saved = session.get(url, params={'id': args.model_id}, timeout=30)
        saved.raise_for_status()
        for key in ('params', 'meta', 'base_model_id', 'name', 'access_grants', 'is_active'):
            if key in configured:
                assert saved.json()[key] == configured[key], f'Unexpected saved field: {key}'
    print(json.dumps({'model': args.model_id, 'applied': args.apply, 'backup': str(args.backup),
                      'changed_fields': ['params.custom_params.tools', 'params.custom_params.tool_choice',
                                         'params.system', 'meta.capabilities.web_search']}))


if __name__ == '__main__':
    main()
