import hashlib
import logging
import os
import re
import shutil
import subprocess
from io import StringIO
from typing import Optional

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from .utils import resolve_asset_path, get_user_app_data_path
from .platform import keyboard

COMMAND_ACTIONS = ('hotkey', 'type', 'run')


class CommandsFileChangedError(Exception):
    pass


def _create_round_trip_yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096
    return yaml


def normalize_spoken_text(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def find_matching_commands(commands: list, text: str) -> list:
    normalized = normalize_spoken_text(text)
    ordered_commands = sorted(commands, key=lambda command: len(str(command.get('trigger', ''))), reverse=True)
    return [command for command in ordered_commands
            if command.get('trigger') and str(command['trigger']).lower() in normalized]


def command_to_entry(command, source_index=None) -> dict:
    action = next((key for key in COMMAND_ACTIONS if key in command), 'hotkey')
    return {
        'trigger': str(command.get('trigger') or ''),
        'action': action,
        'value': str(command.get(action) or ''),
        'enabled': command.get('enabled', True) is not False,
        'source_index': source_index,
    }


def entry_to_command(entry: dict) -> dict:
    return {'trigger': entry['trigger'], entry['action']: entry['value']}


def find_entry_problems(entries: list) -> list:
    problems = []
    used_triggers = set()
    for position, entry in enumerate(entries, start=1):
        trigger = entry['trigger'].strip()
        label = f"'{trigger}'" if trigger else f"Command {position}"
        if not trigger:
            problems.append(f"{label}: trigger is empty")
        if not entry['value'].strip():
            problems.append(f"{label}: {entry['action']} value is empty")
        if entry['action'] == 'hotkey':
            unsendable_keys = [key.strip() for key in entry['value'].split('+') if key.strip() and not keyboard.can_send_key(key.strip())]
            if unsendable_keys:
                problems.append(f"{label}: key {', '.join(unsendable_keys)} cannot be sent")
        normalized_trigger = trigger.lower()
        if normalized_trigger in used_triggers:
            problems.append(f"{label}: trigger is used twice")
        if normalized_trigger:
            used_triggers.add(normalized_trigger)
    return problems


class VoiceCommandManager:
    def __init__(self, enabled=True, clipboard_manager=None, log_transcriptions=False):
        self.enabled = enabled
        self.clipboard_manager = clipboard_manager
        self.log_transcriptions = log_transcriptions
        self.logger = logging.getLogger(__name__)
        self.commands = []
        self._loaded_entries_file_hash = None

        if not self.enabled:
            self.logger.info("Voice commands disabled by configuration")
            return

        defaults_path = resolve_asset_path("commands.defaults.yaml")
        self.commands_path = os.path.join(get_user_app_data_path(), "commands.yaml")

        if not os.path.exists(self.commands_path):
            shutil.copy2(defaults_path, self.commands_path)
            self.logger.info(f"Created user commands file from defaults: {self.commands_path}")

        self.reload()

    def _load_commands_document(self):
        try:
            with open(self.commands_path, 'r', encoding='utf-8') as f:
                document = _create_round_trip_yaml().load(f)
        except Exception as e:
            self.logger.error(f"Failed to parse {self.commands_path}: {e}")
            raise
        if document is None:
            document = CommentedMap()
        if not isinstance(document.get('commands'), CommentedSeq):
            document['commands'] = CommentedSeq()
        return document

    def reload(self):
        self._apply_commands(self._load_commands_document()['commands'])

    def _apply_commands(self, raw_commands: list):
        commands = self._validate_commands(raw_commands)
        commands.sort(key=lambda cmd: len(cmd.get('trigger', '')), reverse=True)
        self.commands = commands
        self.logger.info(f"Loaded {len(commands)} voice commands")

    def _validate_commands(self, raw_commands: list) -> list:
        valid = []
        for i, cmd in enumerate(raw_commands):
            if not isinstance(cmd, dict):
                self.logger.warning(f"Command {i}: not a mapping, skipping")
                continue

            trigger = cmd.get('trigger', '')
            action_count = sum(1 for key in COMMAND_ACTIONS if key in cmd)

            if not trigger:
                self.logger.warning(f"Command {i}: missing trigger, skipping")
                continue

            if action_count != 1:
                self.logger.warning(f"Command '{trigger}': needs exactly one of 'run', 'hotkey', or 'type', skipping")
                continue

            if cmd.get('enabled', True) is False:
                continue

            valid.append({key: (str(value) if isinstance(value, str) else value) for key, value in cmd.items()})
        return valid

    def _read_commands_file_hash(self) -> str:
        with open(self.commands_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def load_command_entries(self) -> list:
        self._loaded_entries_file_hash = self._read_commands_file_hash()
        raw_commands = self._load_commands_document()['commands']
        return [command_to_entry(command, index) for index, command in enumerate(raw_commands) if isinstance(command, dict)]

    def save_command_entries(self, entries: list):
        if self._read_commands_file_hash() != self._loaded_entries_file_hash:
            raise CommandsFileChangedError("commands.yaml changed on disk since the window opened; close and reopen the window to load it")
        document = self._load_commands_document()
        command_sequence = document['commands']
        original_commands = list(command_sequence)
        original_item_comments = dict(command_sequence.ca.items)

        saved_commands = []
        saved_item_comments = {}
        for new_index, entry in enumerate(entries):
            source_index = entry.get('source_index')
            reuse_source = (source_index is not None and source_index < len(original_commands)
                            and isinstance(original_commands[source_index], CommentedMap))
            command = original_commands[source_index] if reuse_source else CommentedMap()
            if reuse_source and source_index in original_item_comments:
                saved_item_comments[new_index] = original_item_comments[source_index]
            self._write_entry_into_command(command, entry)
            saved_commands.append(command)

        del command_sequence[:]
        command_sequence.extend(saved_commands)
        command_sequence.ca.items.clear()
        command_sequence.ca.items.update(saved_item_comments)

        body = StringIO()
        _create_round_trip_yaml().dump(document, body)
        temporary_path = f"{self.commands_path}.tmp"
        with open(temporary_path, 'w', encoding='utf-8') as f:
            f.write(body.getvalue())
        os.replace(temporary_path, self.commands_path)
        self._loaded_entries_file_hash = self._read_commands_file_hash()

        self._apply_commands(command_sequence)

    def _write_entry_into_command(self, command: CommentedMap, entry: dict):
        trigger = entry['trigger'].strip()
        if command.get('trigger') != trigger:
            command['trigger'] = trigger
        for action in COMMAND_ACTIONS:
            if action != entry['action'] and action in command:
                del command[action]
        if command.get(entry['action']) != entry['value']:
            command[entry['action']] = entry['value']
        if entry['enabled']:
            command.pop('enabled', None)
        else:
            command['enabled'] = False

    def match_command(self, text: str) -> Optional[dict]:
        matches = find_matching_commands(self.commands, text)
        return matches[0] if matches else None

    def execute_command(self, command: dict, use_auto_enter: bool = False):
        trigger = command.get('trigger', '')

        if 'run' in command:
            self._execute_shell(command['run'], trigger)
        elif 'hotkey' in command:
            self._send_hotkey(command['hotkey'], trigger)
        elif 'type' in command:
            self._deliver_text(command['type'], trigger, use_auto_enter)

    def _execute_shell(self, run_str: str, trigger: str):
        try:
            subprocess.Popen(run_str, shell=True)
            self.logger.info(f"Executed command '{trigger}': {run_str}")
            print(f"   Executed: {trigger}")
        except Exception as e:
            self.logger.error(f"Failed to execute command '{trigger}': {e}")
            print(f"   Failed to execute command: {e}")

    def _send_hotkey(self, hotkey_str: str, trigger: str):
        keys = [k.strip() for k in hotkey_str.lower().split('+')]
        try:
            keyboard.send_hotkey(*keys)
            self.logger.info(f"Sent hotkey '{trigger}': {hotkey_str}")
            print(f"   ✓ Sent hotkey: {trigger} [{hotkey_str}]")
        except Exception as e:
            self.logger.error(f"Failed to send hotkey '{trigger}': {e}")
            print(f"   Failed to send hotkey: {e}")

    def _deliver_text(self, text: str, trigger: str, use_auto_enter: bool = False):
        try:
            if self.clipboard_manager:
                self.clipboard_manager.deliver_transcription(text, use_auto_enter)
                if self.log_transcriptions:
                    self.logger.info(f"Delivered text '{trigger}': {text}")
                else:
                    self.logger.info(f"Delivered text for '{trigger}'")
                print(f"   ✓ Typed: {text}")
            else:
                self.logger.error("No clipboard manager available for type command")
                print(f"   Failed: clipboard manager not available")
        except Exception as e:
            self.logger.error(f"Failed to deliver text '{trigger}': {e}")
            print(f"   Failed to deliver text: {e}")
