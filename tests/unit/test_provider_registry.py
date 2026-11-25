import importlib
import json
import re
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm import cost_calculator
from litellm.llms.registry import ProviderFunction
from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter

_AS_OBJECT: Final = TypeAdapter(object)
COST_FUNCTIONS: Final = {
    name: value
    for name in dir(cost_calculator)
    if isinstance(value := _AS_OBJECT.validate_python(getattr(cost_calculator, name)), ProviderFunction)
}


@pytest.mark.parametrize("name", sorted(COST_FUNCTIONS))
def test_cost_provider_function_loads_the_named_callable(name: str):
    provider_function: Final = COST_FUNCTIONS[name]
    module_path, _, attribute = provider_function.path.partition(":")
    assert provider_function.load() is getattr(importlib.import_module(module_path), attribute)


def test_provider_function_imports_nothing_until_called():
    result: Final = run_child_interpreter(
        "import sys, json\n"
        "from litellm.llms.registry import ProviderFunction\n"
        'mean = ProviderFunction("statistics", "mean")\n'
        "before = 'statistics' in sys.modules\n"
        "value = mean([1, 3])\n"
        "print(json.dumps([before, value]))\n",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == [False, 2]


def test_provider_function_rejects_a_non_callable():
    with pytest.raises(TypeError, match=re.escape("litellm.constants:DEFAULT_MAX_RETRIES is not callable")):
        ProviderFunction("litellm.constants", "DEFAULT_MAX_RETRIES").load()


def test_import_litellm_leaves_realtime_providers_unloaded():
    result: Final = run_child_interpreter(
        "import sys, json, litellm\n"
        "print(json.dumps(sorted(m for m in sys.modules if m == 'litellm.realtime_api.main' "
        "or m.endswith(('.realtime.handler', '.live.handler')))))\n",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == []


def test_realtime_entry_points_resolve_on_first_access():
    result: Final = run_child_interpreter(
        "import json, litellm\n"
        "from litellm import arealtime_calls\n"
        "print(json.dumps([litellm._arealtime.__module__, arealtime_calls.__module__]))\n",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == ["litellm.realtime_api.main"] * 2
