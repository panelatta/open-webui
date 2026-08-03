from open_webui.utils.task import get_task_model_id


def test_title_task_prefers_preset_base_model_without_override():
    models = {
        'preset': {
            'connection_type': 'external',
            'info': {'base_model_id': 'base'},
        },
        'base': {'connection_type': 'external'},
    }

    assert (
        get_task_model_id(
            'preset',
            '',
            '',
            models,
            prefer_base_model=True,
        )
        == 'base'
    )


def test_explicit_external_task_model_is_preserved():
    models = {
        'preset': {
            'connection_type': 'external',
            'info': {'base_model_id': 'base'},
        },
        'base': {'connection_type': 'external'},
        'task-preset': {
            'connection_type': 'external',
            'info': {'base_model_id': 'base'},
        },
    }

    assert (
        get_task_model_id(
            'preset',
            '',
            'task-preset',
            models,
            prefer_base_model=True,
        )
        == 'task-preset'
    )


def test_missing_base_model_keeps_default_model():
    models = {
        'preset': {
            'connection_type': 'external',
            'info': {'base_model_id': 'missing'},
        }
    }

    assert (
        get_task_model_id(
            'preset',
            '',
            '',
            models,
            prefer_base_model=True,
        )
        == 'preset'
    )


def test_other_tasks_keep_existing_model_selection_behavior():
    models = {
        'preset': {
            'connection_type': 'external',
            'info': {'base_model_id': 'base'},
        },
        'base': {'connection_type': 'external'},
    }

    assert get_task_model_id('preset', '', '', models) == 'preset'
