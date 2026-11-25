import ast
import inspect
import json
import os
from collections.abc import Iterator
from typing import Final

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm import cost_calculator
from litellm._lazy_imports_registry import FIRST_CALL_MODULES
from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter

_LAZY_FLAG: Final = "LITELLM_DISABLE_LAZY_LOADING"
_REPORT_LOADED: Final = (
    "import sys, json, litellm\n"
    "from litellm._lazy_imports import _LITELLM_FACTORIES\n"
    "from litellm._lazy_imports_registry import FIRST_CALL_MODULES, LITELLM_ATTRIBUTES, UTILS_ATTRIBUTES\n"
    "print(json.dumps({\n"
    "    'litellm': sorted(n for n in [*LITELLM_ATTRIBUTES, *_LITELLM_FACTORIES] if n not in vars(litellm)),\n"
    "    'utils': sorted(n for n in UTILS_ATTRIBUTES if n not in vars(litellm.utils)),\n"
    "    'first_call': sorted(m for m in FIRST_CALL_MODULES if m not in sys.modules),\n"
    "    'realtime': 'litellm.realtime_api.main' in sys.modules,\n"
    "}))\n"
)


class _Report(TypedDict):
    litellm: ReadOnly[tuple[str, ...]]
    utils: ReadOnly[tuple[str, ...]]
    first_call: ReadOnly[tuple[str, ...]]
    realtime: ReadOnly[bool]


_REPORT: Final = TypeAdapter(_Report)


def _loaded_report(flag: str | None) -> _Report:
    environment: Final = {name: value for name, value in os.environ.items() if name != _LAZY_FLAG}
    if flag is not None:
        environment[_LAZY_FLAG] = flag
    result: Final = run_child_interpreter(_REPORT_LOADED, env=environment, timeout=180)
    assert result.returncode == 0, result.stderr
    return _REPORT.validate_json(result.stdout.splitlines()[-1])


def test_import_litellm_leaves_provider_cost_and_realtime_modules_unloaded():
    report: Final = _loaded_report(None)
    assert {
        "litellm.llms.anthropic.cost_calculation",
        "litellm.llms.openai.cost_calculation",
        "litellm.llms.vertex_ai.cost_calculator",
    } <= set(report["first_call"])
    assert report["realtime"] is False


def test_disable_lazy_loading_loads_every_lazy_attribute_and_first_call_module():
    assert _loaded_report("1") == {"litellm": (), "utils": (), "first_call": (), "realtime": True}


def _modules_imported_inside_functions(tree: ast.Module) -> Iterator[str]:
    functions: Final = (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    for function in functions:
        for statement in ast.walk(function):
            if isinstance(statement, ast.ImportFrom) and statement.module is not None:
                yield statement.module


def test_first_call_modules_are_the_provider_modules_cost_calculator_imports_inside_functions():
    imported: Final = _modules_imported_inside_functions(ast.parse(inspect.getsource(cost_calculator)))
    assert {module for module in imported if module.startswith("litellm.llms.")} == set(FIRST_CALL_MODULES)


def test_realtime_entry_points_resolve_on_first_access():
    result: Final = run_child_interpreter(
        "import json, litellm\n"
        "from litellm import arealtime_calls\n"
        "print(json.dumps([litellm._arealtime.__module__, arealtime_calls.__module__]))\n",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == ["litellm.realtime_api.main"] * 2
