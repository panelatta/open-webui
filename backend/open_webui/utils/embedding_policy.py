import logging

log = logging.getLogger(__name__)

EMBEDDING_DISABLED_MESSAGE = (
    'Embedding is disabled outside memory routes. Chat files must use OpenAI hosted file uploads; '
    'knowledge, retrieval, file upload processing, tools, and compatibility embedding APIs must not '
    'generate local embeddings in this deployment.'
)


async def disabled_non_memory_embedding_function(*args, **kwargs):
    raise RuntimeError(EMBEDDING_DISABLED_MESSAGE)


def apply_memory_only_embedding_policy(app_state, memory_embedding_function):
    app_state.MEMORY_EMBEDDING_FUNCTION = memory_embedding_function
    app_state.EMBEDDING_FUNCTION = disabled_non_memory_embedding_function
    log.info('Applied memory-only embedding policy')
