import argparse
import importlib.util
import wave
from pathlib import Path

import numpy as np
import pytest

BENCH_PATH = Path(__file__).resolve().parents[1] / "tools" / "bench" / "transcribe_bench.py"
spec = importlib.util.spec_from_file_location("transcribe_bench", BENCH_PATH)
transcribe_bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transcribe_bench)


def test_word_error_rate_counts_edits_and_ignores_punctuation():
    assert transcribe_bench.word_error_rate("Hello, world!", "hello world") == 0.0
    assert transcribe_bench.word_error_rate("one two three four", "one too three") == 0.5


def test_normalization_ignores_apostrophes_and_splits_hyphens():
    assert transcribe_bench.word_error_rate("Don't follow-up", "dont follow up") == 0.0


def write_wav(path, samples, sample_width=2, rate=16000):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(samples)


def test_reads_24_bit_pcm(tmp_path):
    value = int(0.5 * (1 << 23))
    frame = value.to_bytes(3, "little", signed=True)
    write_wav(tmp_path / "clip.wav", frame * 16000, sample_width=3)
    audio = transcribe_bench.read_wav_16k_mono(tmp_path / "clip.wav")
    assert len(audio) == 16000 and np.allclose(audio, 0.5, atol=1e-6)


class CountingEngine:
    def __init__(self):
        self.calls = 0

    def transcribe_audio(self, audio_data, language=None):
        self.calls += 1
        return "hello there world"


def test_benchmark_reports_latency_rtf_and_both_wers(tmp_path):
    write_wav(tmp_path / "a.wav", np.zeros(16000, dtype=np.int16).tobytes())
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    write_wav(tmp_path / "b.wav", np.zeros(16000, dtype=np.int16).tobytes())
    (tmp_path / "b.txt").write_text("hello there world now", encoding="utf-8")
    engine = CountingEngine()

    result = transcribe_bench.run_benchmark(engine, transcribe_bench.load_samples(tmp_path), runs=2)
    first = result["rows"][0]
    assert engine.calls == 6
    assert first["audio_seconds"] == 1.0
    assert first["latency_seconds"] >= 0 and first["real_time_factor"] is not None
    assert first["text"] == "hello there world"
    assert result["mean_wer"] == 0.375
    assert result["corpus_wer"] == round(2 / 6, 4)


def test_missing_values_are_formatted_as_dashes():
    assert transcribe_bench.format_optional(None, ".4f") == "-"
    assert transcribe_bench.format_optional(0.5, ".1f") == "0.5"


def test_runs_must_be_positive_and_threads_non_negative():
    with pytest.raises(argparse.ArgumentTypeError):
        transcribe_bench.positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        transcribe_bench.non_negative_int("-1")
    assert transcribe_bench.non_negative_int("0") == 0


class FakeEngine:
    ENGINE_TYPE = "faster_whisper"

    def __init__(self, model_key="tiny", device="cpu"):
        self.model_key = model_key
        self.device = device


def bench_args(**overrides):
    values = {"model": "tiny", "runtime": None, "onnx_runtime": None, **overrides}
    return argparse.Namespace(**values)


def test_engine_fallbacks_are_reported_as_mismatches():
    assert transcribe_bench.engine_mismatches(FakeEngine(), bench_args(), {"engine_type": "faster_whisper", "device": "cpu"}) == []
    whisper_cpp_fell_back = transcribe_bench.engine_mismatches(FakeEngine(), bench_args(runtime="vulkan"), {"engine_type": "whisper_cpp"})
    assert any("engine_type" in mismatch for mismatch in whisper_cpp_fell_back)
    inactive_cuda = transcribe_bench.engine_mismatches(FakeEngine(), bench_args(runtime="cuda"), {"engine_type": "faster_whisper", "device": "cpu"})
    assert any("not active" in mismatch for mismatch in inactive_cuda)
    other_model = transcribe_bench.engine_mismatches(FakeEngine(model_key="tiny"), bench_args(model="base"), {"engine_type": "faster_whisper", "device": "cpu"})
    assert any("model" in mismatch for mismatch in other_model)


class FakeOnnxEngine:
    ENGINE_TYPE = "onnx_asr"
    model_key = "qwen3-asr-1.7b-int8"
    onnx_runtime = "onnx-directml"
    device = "cpu"


def test_onnx_gpu_request_that_runs_on_the_cpu_is_a_mismatch():
    expected = {"engine_type": "onnx_asr", "onnx_runtime": "onnx-directml", "device": "gpu"}
    mismatches = transcribe_bench.engine_mismatches(FakeOnnxEngine(), bench_args(model="qwen3-asr-1.7b-int8", onnx_runtime="onnx-directml"), expected)
    assert any("device [cpu] instead of [gpu]" in mismatch for mismatch in mismatches)
    assert transcribe_bench.describe_engine(FakeOnnxEngine()) == "onnx_asr [qwen3-asr-1.7b-int8] on onnx-directml / cpu"
