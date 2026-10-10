import importlib.util
import wave
from pathlib import Path

import numpy as np

BENCH_PATH = Path(__file__).resolve().parents[1] / "tools" / "bench" / "transcribe_bench.py"
spec = importlib.util.spec_from_file_location("transcribe_bench", BENCH_PATH)
transcribe_bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transcribe_bench)


def test_word_error_rate_counts_edits_and_ignores_punctuation():
    assert transcribe_bench.word_error_rate("Hello, world!", "hello world") == 0.0
    assert transcribe_bench.word_error_rate("one two three four", "one too three") == 0.5


def test_benchmark_reports_latency_and_wer(tmp_path):
    audio = (np.zeros(16000) * 32767).astype(np.int16)
    with wave.open(str(tmp_path / "clip.wav"), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio.tobytes())
    (tmp_path / "clip.txt").write_text("hello world", encoding="utf-8")

    class EchoEngine:
        def transcribe_audio(self, audio_data):
            return "hello there world"

    result = transcribe_bench.run_benchmark(EchoEngine(), transcribe_bench.load_samples(tmp_path), runs=2)
    assert result["rows"][0]["audio_seconds"] == 1.0
    assert result["mean_wer"] == 0.5
