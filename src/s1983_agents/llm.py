"""Model access behind a one-method protocol.

Every agent asks for a Pydantic object back (structured output), never free
text, so the graph can route on fields like `verdict`. Tests swap in a fake
that satisfies the same protocol, so the whole graph runs offline.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from .config import Role, Settings

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    def structured(self, role: Role, schema: type[T], system: str, user: str) -> T: ...


class LangChainLLM:
    """Provider-agnostic: each role's model string is 'provider:model'."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: dict[tuple[Role, type[BaseModel]], object] = {}

    def _runnable(self, role: Role, schema: type[T]):
        key = (role, schema)
        if key not in self._cache:
            model = init_chat_model(
                self._settings.model_for(role),
                temperature=self._settings.temperature,
                max_tokens=16_000,
            )
            self._cache[key] = model.with_structured_output(schema)
        return self._cache[key]

    def structured(self, role: Role, schema: type[T], system: str, user: str) -> T:
        result = self._runnable(role, schema).invoke([("system", system), ("user", user)])
        if not isinstance(result, schema):  # some providers return dicts
            result = schema.model_validate(result)
        return result
