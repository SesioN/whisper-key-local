import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SAMPLE_RATE = 16000
PUBLIC_SAMPLES = {
    "jfk": ("https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav",
            "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."),
}


def normalize_words(text: str) -> list:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    previous_row = list(range(len(hypothesis_words) + 1))
    for row_index, reference_word in enumerate(reference_words, start=1):
        current_row = [row_index]
        for column_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution = previous_row[column_index - 1] + (reference_word != hypothesis_word)
            current_row.append(min(previous_row[column_index] + 1, current_row[column_index - 1] + 1, substitution))
        previous_row = current_row
    return previous_row[-1] / len(reference_words)


def read_wav_16k_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError(f"{path.name}: only 16-bit PCM WAV files are supported")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channel_count > 1:
        audio = audio.reshape(-1, channel_count).mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        import soxr
        audio = soxr.resample(audio, sample_rate, SAMPLE_RATE).astype(np.float32)
    return audio


def load_samples(sample_dir: Path) -> list:
    samples = []
    for wav_path in sorted(sample_dir.glob("*.wav")):
        reference_path = wav_path.with_suffix(".txt")
        reference = reference_path.read_text(encoding="utf-8").strip() if reference_path.is_file() else None
        samples.append((wav_path.stem, read_wav_16k_mono(wav_path), reference))
    return samples


def download_public_samples(sample_dir: Path):
    sample_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, reference) in PUBLIC_SAMPLES.items():
        wav_path = sample_dir / f"{name}.wav"
        if not wav_path.is_file():
            urllib.request.urlretrieve(url, wav_path)
        wav_path.with_suffix(".txt").write_text(reference, encoding="utf-8")


def build_engine(args):
    from whisper_key.config_manager import ConfigManager
    from whisper_key.model_registry import ModelRegistry
    from whisper_key.runtime_options import with_whisper_cpp_paths
    from whisper_key.whisper_engine import create_whisper_engine

    config_manager = ConfigManager(use_user_settings=False)
    whisper_config = with_whisper_cpp_paths(config_manager.get_whisper_config())
    overrides = {
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "language": args.language,
        "onnx_runtime": args.onnx_runtime,
        "cpu_threads": args.cpu_threads,
        "without_timestamps": args.without_timestamps,
    }
    whisper_config.update({key: value for key, value in overrides.items() if value is not None})
    registry = ModelRegistry(whisper_models_config=whisper_config.get("models", {}))
    engine = create_whisper_engine(args.engine, whisper_config, None, registry)
    engine.warm_up()
    return engine


def run_benchmark(engine, samples: list, runs: int) -> dict:
    rows = []
    for name, audio, reference in samples:
        timings = []
        text = ""
        for _ in range(runs):
            start = time.perf_counter()
            text = engine.transcribe_audio(audio) or ""
            timings.append(time.perf_counter() - start)
        audio_seconds = len(audio) / SAMPLE_RATE
        latency = statistics.median(timings)
        rows.append({
            "sample": name,
            "audio_seconds": round(audio_seconds, 2),
            "latency_seconds": round(latency, 3),
            "real_time_factor": round(latency / audio_seconds, 4) if audio_seconds else None,
            "wer": round(word_error_rate(reference, text), 4) if reference is not None else None,
            "text": text,
        })
    scored = [row["wer"] for row in rows if row["wer"] is not None]
    return {
        "rows": rows,
        "mean_wer": round(statistics.mean(scored), 4) if scored else None,
        "total_latency_seconds": round(sum(row["latency_seconds"] for row in rows), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Whisper Key transcription engines on WAV files with reference .txt files")
    parser.add_argument("samples", type=Path, help="Folder with *.wav files and matching *.txt references")
    parser.add_argument("--download-public-samples", action="store_true")
    parser.add_argument("--engine", default="faster_whisper", choices=("faster_whisper", "whisper_cpp", "onnx_asr"))
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--device")
    parser.add_argument("--compute-type")
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--language")
    parser.add_argument("--onnx-runtime")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--without-timestamps", choices=("auto", "always", "never"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.download_public_samples:
        download_public_samples(args.samples)
    samples = load_samples(args.samples)
    if not samples:
        parser.error(f"No WAV files found in {args.samples}")

    result = run_benchmark(build_engine(args), samples, args.runs)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    for row in result["rows"]:
        wer = "-" if row["wer"] is None else f"{row['wer']:.3f}"
        print(f"{row['sample']:<24} {row['audio_seconds']:>7.2f}s  latency {row['latency_seconds']:>7.3f}s  "
              f"RTF {row['real_time_factor']:.4f}  WER {wer}")
    print(f"mean WER {result['mean_wer']}  total latency {result['total_latency_seconds']}s")


if __name__ == "__main__":
    main()
