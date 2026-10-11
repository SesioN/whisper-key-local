import ast
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

PLATFORM_DIR = Path(__file__).resolve().parents[1] / "src" / "whisper_key" / "platform"


def platform_module_names() -> list:
    return sorted(path.stem for path in (PLATFORM_DIR / "windows").glob("*.py") if path.stem != "__init__")


def defined_names(module_path: Path) -> set:
    names = set()
    for node in ast.walk(ast.parse(module_path.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    return names


def _install_platform_stub():
    platform_stub = types.ModuleType('whisper_key.platform')
    platform_stub.PLATFORM = 'test'
    platform_stub.IS_MACOS = False
    platform_stub.IS_WINDOWS = False
    for module_name in platform_module_names():
        api = defined_names(PLATFORM_DIR / "windows" / f"{module_name}.py") | defined_names(PLATFORM_DIR / "macos" / f"{module_name}.py")
        module_stub = MagicMock(name=f'whisper_key.platform.{module_name}', spec=sorted(api))
        setattr(platform_stub, module_name, module_stub)
        sys.modules[f'whisper_key.platform.{module_name}'] = module_stub
    platform_stub.cpu.physical_core_count.return_value = 4
    platform_stub.clipboard.snapshot.return_value = None
    platform_stub.clipboard.change_count.return_value = 0
    sys.modules['whisper_key.platform'] = platform_stub


def _install_optional_stub(module_name, expected_errors):
    try:
        __import__(module_name)
    except expected_errors:
        sys.modules[module_name] = MagicMock(name=module_name)


_install_platform_stub()
_install_optional_stub('sounddevice', (ImportError, OSError))
_install_optional_stub('pystray', (ImportError, OSError, Exception))
