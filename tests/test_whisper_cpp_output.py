from whisper_key.whisper_cpp_engine import WhisperCppEngine


def test_clean_output_removes_non_speech_tags_and_joins_lines():
    assert WhisperCppEngine._clean_output(" Hello there.\n[BLANK_AUDIO]\n (music) General Kenobi.\n") == "Hello there. General Kenobi."
