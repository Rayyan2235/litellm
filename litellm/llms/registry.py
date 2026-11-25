"""Provider functions named by import path, imported on their first call."""

import importlib
from collections.abc import Callable
from typing import Final, final

from pydantic import TypeAdapter

_AS_OBJECT: Final = TypeAdapter(object)


@final
class ProviderFunction:
    __slots__ = ("_attribute", "_loaded", "_module")

    def __init__(self, module: str, attribute: str) -> None:
        self._module: Final = module
        self._attribute: Final = attribute
        self._loaded: Callable[..., object] | None = None

    @property
    def path(self) -> str:
        return f"{self._module}:{self._attribute}"

    def load(self) -> Callable[..., object]:
        if self._loaded is None:
            loaded: Final = _AS_OBJECT.validate_python(getattr(importlib.import_module(self._module), self._attribute))
            if not callable(loaded):
                raise TypeError(f"{self.path} is not callable")
            self._loaded = loaded
        return self._loaded

    def __call__(
        self,
        *args: object,
        **kwargs: object,  # kwargs-ok: forwarded unchanged to the provider function
    ) -> object:
        return self.load()(*args, **kwargs)

    def __repr__(self) -> str:
        return f"ProviderFunction({self.path!r})"
