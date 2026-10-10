import sys
import types
from unittest.mock import MagicMock

PLATFORM_MODULES = ('instance_lock', 'keyboard', 'hotkeys', 'paths', 'app', 'permissions', 'icons', 'gpu', 'console',
                    'window_style', 'dialogs', 'theme', 'child_processes', 'autostart', 'media', 'clipboard')


def _install_platform_stub():
    platform_stub = types.ModuleType('whisper_key.platform')
    platform_stub.PLATFORM = 'test'
    platform_stub.IS_MACOS = False
    platform_stub.IS_WINDOWS = False
    for module_name in PLATFORM_MODULES:
        module_stub = MagicMock(name=f'whisper_key.platform.{module_name}')
        setattr(platform_stub, module_name, module_stub)
        sys.modules[f'whisper_key.platform.{module_name}'] = module_stub
    sys.modules['whisper_key.platform'] = platform_stub


def _install_optional_stub(module_name):
    try:
        __import__(module_name)
    except Exception:
        sys.modules[module_name] = MagicMock(name=module_name)


_install_platform_stub()
sys.modules['pystray'] = MagicMock(name='pystray')
_install_optional_stub('sounddevice')
