"""Model-owned reasoning presets. Clients send IDs, never parameter mappings."""

import re
from copy import deepcopy
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

PATHS = {
    'reasoning_effort',
    'reasoning.effort',
    'reasoning.max_tokens',
    'reasoning.enabled',
    'reasoning.exclude',
    'thinking.type',
    'thinking.budget_tokens',
    'output_config.effort',
}


def validate_binding(path, value):
    if path not in PATHS:
        raise ValueError(f'Unsupported reasoning field: {path}')
    if value is None:
        return
    if path.endswith(('max_tokens', 'budget_tokens')):
        if type(value) is not int or not 1 <= value <= 1000000:
            raise ValueError('Reasoning budget must be an integer between 1 and 1000000')
    elif path.endswith(('enabled', 'exclude')):
        if type(value) is not bool:
            raise ValueError('Reasoning flag must be a boolean')
    elif not isinstance(value, str) or not value or len(value) > 64:
        raise ValueError('Reasoning value must be a nonempty string of at most 64 characters')


class ReasoningLevel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_-]+$')
    labels: dict[str, str] = Field(default_factory=dict, max_length=64)
    value: str | int | bool | None
    bindings: dict = Field(default_factory=dict, max_length=8)

    @model_validator(mode='after')
    def check_level(self):
        if any(not k or len(k) > 35 or not v.strip() or len(v) > 120 for k, v in self.labels.items()):
            raise ValueError('Locale labels must be nonempty and at most 120 characters')
        for path, value in self.bindings.items():
            validate_binding(path, value)
        return self


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enabled: bool = True
    field: str = 'reasoning_effort'
    default_id: str = ''
    options: list[ReasoningLevel] = Field(default_factory=list, max_length=32)

    @model_validator(mode='after')
    def check_config(self):
        if self.field not in PATHS:
            raise ValueError('Unsupported reasoning field')
        ids = [option.id for option in self.options]
        if len(ids) != len(set(ids)) or (self.default_id and self.default_id not in ids):
            raise ValueError('Reasoning IDs must be unique and include the default ID')
        if self.enabled and not ids:
            raise ValueError('Add at least one reasoning level')
        for option in self.options:
            validate_binding(self.field, option.value)
            if self.field in option.bindings:
                raise ValueError('The main reasoning field cannot also be an additional binding')
        return self


def model_chain(model_id, infos):
    """Resolve aliases with an explicit cycle/depth guard; nearest config wins."""
    config = None
    seen = set()
    while model_id in infos:
        if model_id in seen or len(seen) >= 32:
            raise ValueError('Cyclic or excessively deep base model chain')
        seen.add(model_id)
        info = infos[model_id]
        candidate = (info.get('meta') or {}).get('reasoning_effort_config')
        if config is None and candidate is not None:
            config = ReasoningConfig.model_validate(candidate).model_dump()
        base = info.get('base_model_id')
        if not base:
            break
        model_id = base
    return model_id, config


def default_config(model_id, url=''):
    name = model_id.lower().split('/')[-1]
    is_openrouter = urlparse(url).hostname == 'openrouter.ai'
    # Do not guess support for chat/audio/image/non-reasoning variants.
    if any(part in name for part in ('chat', 'audio', 'image', 'search', 'spark')):
        return None
    if re.match(r'^(gpt-[5-9](?:[.\-]|$)|o[134](?:-|$)|gpt-oss-)', name):
        values = ['low', 'medium', 'high']
        if '-pro' in name:
            values = ['high']
    elif name.startswith('claude-') and is_openrouter:
        values = ['low', 'medium', 'high']
    else:
        return None
    names = {'low': '轻量', 'medium': '标准', 'high': '深入'}
    return {
        'enabled': True,
        'field': 'reasoning.effort' if is_openrouter else 'reasoning_effort',
        'default_id': '',
        'options': [
            {'id': v, 'labels': {'en-US': v.capitalize(), 'zh-CN': names[v]}, 'value': v, 'bindings': {}}
            for v in values
        ],
    }


def set_path(payload, path, value):
    keys = path.split('.')
    current = payload
    for key in keys[:-1]:
        if not isinstance(current.get(key), dict):
            if value is None:
                return
            current[key] = {}
        current = current[key]
    if value is None:
        current.pop(keys[-1], None)
    else:
        current[keys[-1]] = deepcopy(value)


def apply_reasoning_level(payload, config, selected=None):
    if config is None or not config.get('enabled'):
        if selected:
            raise ValueError('Reasoning levels are not enabled for this model; reload the model list')
        return payload
    config = ReasoningConfig.model_validate(config).model_dump()
    level_id = config['default_id'] if selected is None else selected
    if not level_id:
        return payload  # Model default preserves existing advanced params.
    option = next((o for o in config['options'] if o['id'] == level_id), None)
    if option is None:
        raise ValueError('Reasoning level no longer exists; choose a current level')
    # Remove competing effort aliases, while preserving summary and other siblings.
    for path in ('reasoning_effort', 'reasoning.effort', 'output_config.effort'):
        set_path(payload, path, None)
    # Clear all fields owned by this configuration so switching leaves no stale bindings.
    for path in {config['field']} | {p for o in config['options'] for p in o['bindings']}:
        set_path(payload, path, None)
    for path, value in option['bindings'].items():
        set_path(payload, path, value)
    set_path(payload, config['field'], option['value'])
    validate_thinking_budget(payload)
    return payload


def validate_thinking_budget(payload):
    thinking = payload.get('thinking') or {}
    budget = thinking.get('budget_tokens')
    if budget is not None:
        if thinking.get('type') != 'enabled' or budget < 1024:
            raise ValueError('Claude budget requires thinking.type=enabled and at least 1024 tokens')
        limit = payload.get('max_tokens', payload.get('max_completion_tokens'))
        if limit is not None and budget >= limit:
            raise ValueError('Thinking budget must be smaller than max_tokens')
