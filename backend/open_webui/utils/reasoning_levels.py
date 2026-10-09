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


def advertised_levels(model):
    """Return provider effort IDs, or None when no discrete capability is advertised.

    An explicit empty/invalid list disables automatic presets rather than inventing
    three levels. Values remain provider-owned; labels never change request values.
    """
    thinking = model.get('thinking')
    if isinstance(thinking, dict) and 'levels' in thinking:
        levels = thinking['levels']
    elif 'supported_reasoning_levels' in model:
        levels = model['supported_reasoning_levels']
        if isinstance(levels, list):
            levels = [item.get('effort') if isinstance(item, dict) else item for item in levels]
    elif isinstance(model.get('reasoning'), dict) and 'supported_efforts' in model['reasoning']:
        levels = model['reasoning']['supported_efforts']
    else:
        return None
    if not isinstance(levels, list) or len(levels) > 32:
        return []
    if any(not isinstance(v, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', v) for v in levels):
        return []
    return list(dict.fromkeys(levels))


def default_config(model_id, url='', model=None):
    name = model_id.lower().split('/')[-1]
    is_openrouter = urlparse(url).hostname == 'openrouter.ai'
    values = advertised_levels(model or {})
    if values is None:
        # Only use legacy conservative defaults when the provider has no metadata.
        if any(part in name for part in ('chat', 'audio', 'image', 'search', 'spark')):
            return None
        if re.match(r'^(gpt-[5-9](?:[.\-]|$)|o[134](?:-|$)|gpt-oss-)', name):
            values = ['high'] if '-pro' in name else ['low', 'medium', 'high']
        elif name.startswith('claude-') and is_openrouter:
            values = ['low', 'medium', 'high']
        else:
            return None
    if not values:
        return None
    names = {
        'none': ('None', '无'),
        'minimal': ('Minimal', '极低'),
        'low': ('Low', '轻量'),
        'medium': ('Medium', '标准'),
        'high': ('High', '深入'),
        'xhigh': ('Extra high', '超高'),
        'max': ('Max', '最大'),
        'ultra': ('Ultra', '极致'),
    }
    return {
        'enabled': True,
        'field': 'reasoning.effort' if is_openrouter else 'reasoning_effort',
        'default_id': '',
        'options': [
            {'id': v, 'labels': {'en-US': names.get(v, (v, v))[0], 'zh-CN': names.get(v, (v, v))[1]}, 'value': v, 'bindings': {}}
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


def apply_client_reasoning_effort(payload, config, model_id, model, url, requested):
    """Translate legacy client effort using provider-owned levels, never name guesses.

    Call only for an explicit client reasoning_effort without a web preset ID.
    Saved presets provide protocol/bindings, not the list of valid effort values.
    """
    payload.pop('reasoning_effort', None)
    if config is not None and not config.get('enabled'):
        return payload

    levels = advertised_levels(model)
    if levels is None:
        # Catalogs without a reasoning capability must not acquire one from a
        # client default (e.g. switching from GPT-5 to GPT-4o in Open Relay).
        meta = (model.get('info') or {}).get('meta') or {}
        capabilities = meta.get('capabilities') or {}
        supported = model.get('supported_parameters') or []
        name = model_id.lower().split('/')[-1]
        reasoning_model = (
            capabilities.get('reasoning') is True
            or any(p in supported for p in ('reasoning', 'reasoning_effort'))
            or re.match(r'^(gpt-[5-9](?:[.\-]|$)|o[134](?:-|$)|claude-|.*glm)', name)
            or model.get('name', '').startswith('abliterated-model')
        )
        if reasoning_model:
            raise ValueError('Upstream reasoning levels are unavailable; refresh the model list and retry')
        return payload
    if not levels:
        return payload

    if not isinstance(requested, str):
        raise ValueError('reasoning_effort must be a string')  # noqa: TRY004 - router maps validation to HTTP 400
    normalized = requested.strip().casefold()
    value = next((v for v in levels if v.casefold() == normalized), None)
    options = (config or {}).get('options', [])
    if value is None:
        # Allow named web presets only when their actual value is advertised.
        option = next((o for o in options if o['id'].casefold() == normalized and o['value'] in levels), None)
        if option is not None:
            value = option['value']
    if value is None:
        name = model_id.lower().split('/')[-1]
        provider_name = str(model.get('name') or model.get('display_name') or '').lower()
        if 'glm' in name or provider_name.startswith('abliterated-model-large'):
            preferred = ('max',)
        elif re.match(r'^(gpt-|o[134](?:-|$)|claude-)', name):
            preferred = ('xhigh', 'high')
        else:
            preferred = ()
        value = next((v for preferred_value in preferred for v in levels if v.casefold() == preferred_value), None)
        if value is None:
            raise ValueError('Unrecognized reasoning_effort; upstream supports: ' + ', '.join(levels))

    # Reuse saved bindings/field when they correspond to the provider value.
    option = next((o for o in options if o['value'] == value), None)
    field = (config or {}).get('field') or (
        'reasoning.effort' if urlparse(url).hostname == 'openrouter.ai' else 'reasoning_effort'
    )
    if field not in ('reasoning_effort', 'reasoning.effort', 'output_config.effort'):
        raise ValueError('This model uses thinking budgets; select an explicit web reasoning preset')
    selected = deepcopy(option) if option is not None else {
        'id': value, 'value': value, 'labels': {}, 'bindings': {},
    }
    # Clear bindings owned by all saved presets before applying one choice.
    # Do not append to a full saved option list when the upstream adds a value.
    for path in {field} | {p for o in options for p in o.get('bindings', {})}:
        set_path(payload, path, None)
    effective = {'enabled': True, 'field': field, 'default_id': '', 'options': [selected]}
    return apply_reasoning_level(payload, effective, selected['id'])
