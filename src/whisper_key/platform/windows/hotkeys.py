import time

from global_hotkeys import register_hotkeys, start_checking_hotkeys, stop_checking_hotkeys, clear_hotkeys
from global_hotkeys.keycodes import vk_key_names

# global-hotkeys library expects: 'control + window + shift' format
KEY_MAP = {
    'ctrl': 'control',
    'win': 'window',
    'windows': 'window',
    'cmd': 'window',
    'super': 'window',
    'esc': 'escape',
}

CHECKER_THREAD_EXIT_SECONDS = 0.1

MODIFIER_NAMES_BY_VIRTUAL_KEY = {
    0x10: 'shift', 0xA0: 'shift', 0xA1: 'shift',
    0x11: 'ctrl', 0xA2: 'ctrl', 0xA3: 'ctrl',
    0x12: 'alt', 0xA4: 'alt', 0xA5: 'alt',
    0x5B: 'win', 0x5C: 'win',
}

CONFIG_NAMES_BY_LIBRARY_NAME = {library_name: config_name for config_name, library_name in KEY_MAP.items()
                                if config_name in ('ctrl', 'win', 'esc')}

def _build_key_names_by_virtual_key() -> dict:
    key_names = {}
    for name, virtual_key in vk_key_names.items():
        if len(name) == 1 and not name.isalnum():
            continue
        key_names.setdefault(virtual_key, CONFIG_NAMES_BY_LIBRARY_NAME.get(name, name))
    key_names.update(MODIFIER_NAMES_BY_VIRTUAL_KEY)
    return key_names

KEY_NAMES_BY_VIRTUAL_KEY = _build_key_names_by_virtual_key()

def _normalize_hotkey(hotkey_str: str) -> str:
    keys = hotkey_str.lower().split('+')
    converted = [KEY_MAP.get(k.strip(), k.strip()) for k in keys]
    return ' + '.join(converted)

def key_name_for_virtual_key(virtual_key: int):
    return KEY_NAMES_BY_VIRTUAL_KEY.get(virtual_key)

def register(bindings: list):
    normalized = []
    for binding in bindings:
        hotkey_str = binding[0]
        normalized_binding = [_normalize_hotkey(hotkey_str)] + binding[1:]
        normalized.append(normalized_binding)
    register_hotkeys(normalized)

def start():
    start_checking_hotkeys()

def stop():
    stop_checking_hotkeys()

def clear():
    stop_checking_hotkeys()
    time.sleep(CHECKER_THREAD_EXIT_SECONDS)
    clear_hotkeys()
