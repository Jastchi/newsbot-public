"""Build pipeline agents for the evals without an LLM provider."""

from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

# Agents that fetch their provider when built, which for Ollama needs a
# running server. The clustering agent builds a judge agent of its own.
# The evals call neither title generation nor rewrites.
PROVIDER_FACTORIES = (
    "newsbot.agents.name_validation_agent.get_llm_provider",
    "newsbot.agents.story_clustering_agent.get_llm_provider",
    "newsbot.agents.judge_agent.get_llm_provider",
)


class _NoProvider:
    """Stands in for a provider; any call is a bug in the eval."""

    def chat(self, *_args: object, **_kwargs: object) -> str:
        msg = "this eval must not call an LLM"
        raise RuntimeError(msg)


@contextmanager
def without_llm_provider() -> Generator[None]:
    """Build agents in this block with a provider that cannot chat."""
    with ExitStack() as stack:
        for target in PROVIDER_FACTORIES:
            stack.enter_context(patch(target, lambda _config: _NoProvider()))
        yield
