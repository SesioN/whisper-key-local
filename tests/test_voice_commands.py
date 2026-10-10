from whisper_key.voice_commands import find_entry_problems, find_matching_commands


def commands(*triggers):
    return [{'trigger': trigger, 'hotkey': 'ctrl+c'} for trigger in triggers]


def test_trigger_matches_whole_words_only():
    assert find_matching_commands(commands('go'), "I'm going home") == []
    assert find_matching_commands(commands('go'), 'Go!')[0]['trigger'] == 'go'


def test_trigger_punctuation_is_normalized_like_speech():
    assert find_matching_commands(commands("what's up"), 'Whats up?')
    assert find_matching_commands(commands('hello world'), 'Hello - world.')


def test_longest_trigger_wins():
    matches = find_matching_commands(commands('open', 'open browser'), 'Open browser please')
    assert [match['trigger'] for match in matches] == ['open browser', 'open']


def entry(trigger):
    return {'trigger': trigger, 'action': 'type', 'value': 'x', 'enabled': True}


def test_duplicate_triggers_are_detected_after_normalization():
    assert any('used twice' in problem for problem in find_entry_problems([entry("What's up"), entry('whats up')]))


def test_trigger_without_words_is_a_problem():
    assert any('no words' in problem for problem in find_entry_problems([entry('!!!')]))
