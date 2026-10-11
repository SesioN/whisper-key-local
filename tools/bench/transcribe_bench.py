import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SAMPLE_RATE = 16000
DOWNLOAD_TIMEOUT_SECONDS = 30
PUBLIC_SAMPLES = {
    "jfk": ("https://raw.githubusercontent.com/ggml-org/whisper.cpp/v1.9.4/samples/jfk.wav",
            "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."),
}
PCM_DTYPES = {1: np.uint8, 2: np.int16, 4: np.int32}


def normalize_words(text: str) -> list:
    without_apostrophes = re.sub(r"['’]", "", text.lower())
    return re.sub(r"[^\w\s]", " ", without_apostrophes).split()


def word_edits(reference: str, hypothesis: str) -> tuple:
    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    previous_row = list(range(len(hypothesis_words) + 1))
    for row_index, reference_word in enumerate(reference_words, start=1):
        current_row = [row_index]
        for column_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution = previous_row[column_index - 1] + (reference_word != hypothesis_word)
            current_row.append(min(previous_row[column_index] + 1, current_row[column_index - 1] + 1, substitution))
        previous_row = current_row
    return previous_row[-1], len(reference_words)


def error_rate(edits: int, reference_word_count: int) -> float:
    if not reference_word_count:
        return 0.0 if not edits else 1.0
    return edits / reference_word_count


def word_error_rate(reference: str, hypothesis: str) -> float:
    return error_rate(*word_edits(reference, hypothesis))


def _pcm_to_float(frames: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        widened = (raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16))
        widened = np.where(widened >= 1 << 23, widened - (1 << 24), widened)
        return (widened / float(1 << 23)).astype(np.float32)
    if sample_width not in PCM_DTYPES:
        raise ValueError(f"unsupported sample width of {sample_width} bytes")
    samples = np.frombuffer(frames, dtype=PCM_DTYPES[sample_width]).astype(np.float32)
    if sample_width == 1:
        return (samples - 128.0) / 128.0
    return samples / float(1 << (8 * sample_width - 1))


def read_wav_16k_mono(path: Path) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channel_count = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
    except wave.Error as e:
        raise ValueError(f"{path.name}: only PCM WAV files are supported (8/16/24/32-bit), convert float WAVs first: {e}") from e
    audio = _pcm_to_float(frames, sample_width)
    if channel_count > 1:
        audio = audio.reshape(-1, channel_count).mean(axis=1)
    if sample_rate != SAMPLE_RATE:
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
            download_fd, download_path = tempfile.mkstemp(suffix=".part", dir=sample_dir)
            os.close(download_fd)
            try:
                with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, open(download_path, "wb") as target:
                    target.write(response.read())
                os.replace(download_path, wav_path)
            except OSError as e:
                raise RuntimeError(f"Could not download sample {name} from {url}: {e}") from e
            finally:
                if os.path.exists(download_path):
                    os.unlink(download_path)
        wav_path.with_suffix(".txt").write_text(reference, encoding="utf-8")


def expected_engine(whisper_config: dict, registry) -> dict:
    from whisper_key.model_registry import ONNX_FAMILY
    from whisper_key.runtime_options import ONNX_ASR, apply_onnx_runtime_selection, apply_runtime_selection

    if registry.get_engine_family(whisper_config["model"]) == ONNX_FAMILY:
        selected = apply_onnx_runtime_selection(whisper_config)
        return {"engine_type": ONNX_ASR, "onnx_runtime": selected["onnx_runtime"]}
    selected = apply_runtime_selection(whisper_config)
    expected = {"engine_type": selected["engine_type"]}
    if selected["engine_type"] == "faster_whisper":
        expected["device"] = selected["device"]
    return expected


def engine_mismatches(engine, args, expected: dict) -> list:
    mismatches = []
    if engine.model_key != args.model:
        mismatches.append(f"model [{engine.model_key}] instead of [{args.model}]")
    for attribute, value in expected.items():
        actual = engine.ENGINE_TYPE if attribute == "engine_type" else getattr(engine, attribute, None)
        if actual != value:
            mismatches.append(f"{attribute} [{actual}] instead of [{value}]")
    if args.runtime in ("cuda", "rocm") and expected.get("device") != "cuda":
        mismatches.append(f"runtime [{args.runtime}] is not active, select it in Whisper Key first")
    if args.onnx_runtime and expected.get("onnx_runtime", args.onnx_runtime) != args.onnx_runtime:
        mismatches.append(f"ONNX runtime [{args.onnx_runtime}] is not active, select it in Whisper Key first")
    return mismatches


def build_engine(args):
    from whisper_key.config_manager import ConfigManager
    from whisper_key.main import setup_whisper_engine
    from whisper_key.model_registry import ModelRegistry

    whisper_config = ConfigManager(use_user_settings=False).get_whisper_config()
    overrides = {
        "model": args.model,
        "runtime": args.runtime,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "language": args.language,
        "onnx_runtime": args.onnx_runtime,
        "cpu_threads": args.cpu_threads,
    }
    whisper_config.update({key: value for key, value in overrides.items() if value is not None})
    registry = ModelRegistry(whisper_models_config=whisper_config.get("models", {}))
    if not registry.is_known_model(args.model):
        raise SystemExit(f"Unknown model [{args.model}]; use a key from whisper.models in config.defaults.yaml")
    expected = expected_engine(whisper_config, registry)
    engine = setup_whisper_engine(whisper_config, None, registry)
    mismatches = engine_mismatches(engine, args, expected)
    if mismatches:
        engine.close()
        raise SystemExit(f"The requested setup could not be benchmarked, the app would use {describe_engine(engine)}: " + "; ".join(mismatches))
    return engine


def describe_engine(engine) -> str:
    runtime = getattr(engine, "onnx_runtime", None) or getattr(engine, "device", "")
    return f"{engine.ENGINE_TYPE} [{engine.model_key}] on {runtime}"


def run_benchmark(engine, samples: list, runs: int) -> dict:
    rows = []
    total_edits = 0
    total_reference_words = 0
    for name, audio, reference in samples:
        engine.transcribe_audio(audio)
        timings = []
        text = ""
        for _ in range(runs):
            start = time.perf_counter()
            text = engine.transcribe_audio(audio) or ""
            timings.append(time.perf_counter() - start)
        audio_seconds = len(audio) / SAMPLE_RATE
        latency = statistics.median(timings)
        wer = None
        if reference is not None:
            edits, reference_word_count = word_edits(reference, text)
            total_edits += edits
            total_reference_words += reference_word_count
            wer = round(error_rate(edits, reference_word_count), 4)
        rows.append({
            "sample": name,
            "audio_seconds": round(audio_seconds, 2),
            "latency_seconds": round(latency, 3),
            "real_time_factor": round(latency / audio_seconds, 4) if audio_seconds else None,
            "wer": wer,
            "text": text,
        })
    scored = [row["wer"] for row in rows if row["wer"] is not None]
    return {
        "rows": rows,
        "mean_wer": round(statistics.mean(scored), 4) if scored else None,
        "corpus_wer": round(total_edits / total_reference_words, 4) if total_reference_words else None,
        "total_latency_seconds": round(sum(row["latency_seconds"] for row in rows), 3),
    }


def format_optional(value, pattern: str) -> str:
    return "-" if value is None else format(value, pattern)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 (automatic) or more")
    return number


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper Key transcription the way the app runs it: WAV files plus matching .txt references. "
                    "References are compared case-, punctuation- and apostrophe-insensitively; write numbers the way the models do. "
                    "GPU runtimes are activated from your Whisper Key settings, so select the same runtime in the app first.")
    parser.add_argument("samples", type=Path, help="Folder with *.wav files and matching *.txt references")
    parser.add_argument("--download-public-samples", action="store_true")
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--runtime", choices=("cpu", "cuda", "rocm", "vulkan"))
    parser.add_argument("--compute-type")
    parser.add_argument("--beam-size", type=positive_int)
    parser.add_argument("--language")
    parser.add_argument("--onnx-runtime", choices=("onnx-cpu", "onnx-directml", "onnx-cuda"))
    parser.add_argument("--cpu-threads", type=non_negative_int)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.download_public_samples:
        download_public_samples(args.samples)
    samples = load_samples(args.samples)
    if not samples:
        parser.error(f"No WAV files found in {args.samples}")

    engine = build_engine(args)
    try:
        result = run_benchmark(engine, samples, args.runs)
    finally:
        engine.close()
    result["engine"] = describe_engine(engine)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print(result["engine"])
    for row in result["rows"]:
        print(f"{row['sample']:<24} {row['audio_seconds']:>7.2f}s  latency {row['latency_seconds']:>7.3f}s  "
              f"RTF {format_optional(row['real_time_factor'], '.4f')}  WER {format_optional(row['wer'], '.3f')}")
    print(f"mean WER {format_optional(result['mean_wer'], '.3f')}  corpus WER {format_optional(result['corpus_wer'], '.3f')}  "
          f"total latency {result['total_latency_seconds']}s")


if __name__ == "__main__":
    main()
