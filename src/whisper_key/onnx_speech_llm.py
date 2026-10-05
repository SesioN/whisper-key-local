# Adapted from onnx-asr speech-llm family (MIT, https://github.com/TigreGotico/onnx-asr/pull/3, commit 846e10c)
import json
import re
import typing
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import BaseAsr, Preprocessor, TimestampedResult
from onnx_asr.models.whisper import bytes_to_unicode
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_float32_array

NEG_INF = np.float32(-3.4028235e38)
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|.*\|>\Z|<(?:s|/s|unk|pad)>\Z")
BYTE_FALLBACK_PATTERN = re.compile(r"<0x([0-9A-F]{2})>")


def _post_cnn_length(lengths: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    for _ in range(3):
        lengths = np.where(lengths > 0, (lengths - 1) // 2 + 1, 0)
    return lengths


class SpeechLlm(BaseAsr):
    def __init__(
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        config = typing.cast(dict[str, typing.Any], self.config)

        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._embed_tokens = rt.InferenceSession(model_files["embed_tokens"], **onnx_options)
        self._decoder = rt.InferenceSession(model_files["decoder"], **onnx_options)

        with model_files["vocab"].open("rt", encoding="utf-8") as f:
            tokens: dict[str, int] = json.load(f)
        self._vocab = {id: token for token, id in tokens.items()}
        self._byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        self._tokenizer_type = config.get("tokenizer_type", "byte-level")

        self._prefix_ids = config["prompt_prefix_ids"]
        self._suffix_ids = config["prompt_suffix_ids"]
        self._language_prompt_ids: dict[str, list[int]] = config.get("language_prompt_ids", {})
        self._eos_token_ids = set(config["eos_token_ids"])
        self._text_start_token_id = config.get("text_start_token_id")
        self._max_sequence_length = config.get("max_sequence_length", 512)

        self._normalize_waveform = config.get("normalize_waveform", False)
        self._n_window = config.get("n_window", 50)
        self._n_window_infer = config.get("n_window_infer", 800)
        self._max_frames = config.get("max_frames", 3000)
        self._hop_length = config.get("hop_length", 160)

        self._encoder_inputs = {x.name for x in self._encoder.get_inputs()}
        self._past_names = [x.name for x in self._decoder.get_inputs() if x.name.startswith("past_key_values.")]
        self._present_names = [name.replace("past_key_values.", "present.") for name in self._past_names]
        past_shape = next(x.shape for x in self._decoder.get_inputs() if x.name.startswith("past_key_values."))
        self._num_kv_heads = past_shape[1]
        self._head_dim = past_shape[3]

    @property
    def languages(self) -> list:
        return sorted(name for name in self._language_prompt_ids if name)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"**/encoder{suffix}.onnx",
            "embed_tokens": f"**/embed_tokens{suffix}.onnx",
            "decoder": f"**/decoder{suffix}.onnx",
            "vocab": "vocab.json",
        }

    @property
    def _preprocessor_name(self) -> str:
        config = typing.cast(dict[str, typing.Any], self.config)
        return str(config.get("preprocessor", f"whisper{config.get('features_size', 128)}"))

    def _audio_windows(self, feature_len: int) -> tuple[int, npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        chunk_len = self._n_window * 2
        frames_per_chunk = _post_cnn_length(np.array([chunk_len]))[0]

        padded_frames = -(-feature_len // chunk_len) * chunk_len
        num_chunks = padded_frames // chunk_len
        starts = np.arange(num_chunks) * chunk_len
        chunk_lengths = np.clip(feature_len - starts, 0, chunk_len)
        chunk_out_lengths = _post_cnn_length(chunk_lengths)

        valid_indices = np.concatenate(
            [np.arange(length) + i * frames_per_chunk for i, length in enumerate(chunk_out_lengths)]
        ).astype(np.int64)

        window = int(chunk_out_lengths.max()) * (self._n_window_infer // chunk_len)
        total = int(chunk_out_lengths.sum())
        blocks = [window] * (total // window)
        if total % window:
            blocks.append(total % window)

        bias = np.full((1, 1, total, total), NEG_INF, dtype=np.float32)
        offset = 0
        for size in blocks:
            bias[0, 0, offset : offset + size, offset : offset + size] = 0.0
            offset += size

        return padded_frames, valid_indices, bias

    def _encode(self, features: npt.NDArray[np.float32], feature_len: int) -> npt.NDArray[np.float32]:
        inputs: dict[str, npt.NDArray[typing.Any]]
        if self._encoder_inputs == {"input_features"}:
            inputs = {"input_features": features[None]}
        else:
            padded_frames, valid_indices, bias = self._audio_windows(feature_len)
            inputs = {
                "input_features": features[None, :, :padded_frames],
                "valid_indices": valid_indices,
                "attn_bias": bias,
            }

        (audio_embeds,) = self._encoder.run(["audio_embeds"], inputs)
        assert is_float32_array(audio_embeds)
        return audio_embeds

    def _embed(self, token_ids: list[int]) -> npt.NDArray[np.float32]:
        (embeds,) = self._embed_tokens.run(["inputs_embeds"], {"input_ids": np.array([token_ids], dtype=np.int64)})
        assert is_float32_array(embeds)
        return embeds

    def _empty_state(self) -> dict[str, npt.NDArray[np.float32]]:
        empty = np.zeros((1, self._num_kv_heads, 0, self._head_dim), dtype=np.float32)
        return dict.fromkeys(self._past_names, empty)

    def _decode_step(
        self,
        inputs_embeds: npt.NDArray[np.float32],
        state: dict[str, npt.NDArray[np.float32]],
        past_length: int,
    ) -> tuple[npt.NDArray[np.float32], dict[str, npt.NDArray[np.float32]]]:
        seq_len = inputs_embeds.shape[1]
        total = past_length + seq_len
        bias = np.zeros((1, 1, seq_len, total), dtype=np.float32)
        causal = np.arange(total)[None, :] > (past_length + np.arange(seq_len))[:, None]
        bias[0, 0][causal] = NEG_INF
        position_ids = np.arange(past_length, total, dtype=np.int64)[None]

        outputs = self._decoder.run(
            ["logits", *self._present_names],
            {"inputs_embeds": inputs_embeds, "attn_bias": bias, "position_ids": position_ids, **state},
        )
        logits = outputs[0]
        assert is_float32_array(logits)
        return logits, dict(zip(self._past_names, outputs[1:], strict=True))

    def _generate(self, audio_embeds: npt.NDArray[np.float32], language: str | None) -> list[int]:
        prefix_ids = self._prefix_ids
        if language:
            prompts = self._language_prompt_ids
            prefix_ids = prompts.get(language, prompts.get(language.title(), prefix_ids))

        inputs_embeds = np.concatenate([self._embed(prefix_ids), audio_embeds, self._embed(self._suffix_ids)], axis=1)
        state = self._empty_state()
        past_length = 0
        tokens: list[int] = []

        for _ in range(self._max_sequence_length):
            logits, state = self._decode_step(inputs_embeds, state, past_length)
            past_length += inputs_embeds.shape[1]
            token = int(logits[0, -1].argmax())
            if token in self._eos_token_ids:
                break
            tokens.append(token)
            inputs_embeds = self._embed([token])

        return tokens

    def _decode_tokens(self, tokens: list[int]) -> TimestampedResult:
        if self._text_start_token_id in tokens:
            tokens = tokens[tokens.index(self._text_start_token_id) + 1 :]

        parts = [token for id in tokens if (token := self._vocab[id]) and not SPECIAL_TOKEN_PATTERN.match(token)]

        if self._tokenizer_type == "sentencepiece":
            data = bytearray()
            for token in parts:
                if (byte := BYTE_FALLBACK_PATTERN.fullmatch(token)) is not None:
                    data.append(int(byte[1], 16))
                else:
                    data += token.replace("▁", " ").encode()
        else:
            data = bytearray(self._byte_decoder[char] for char in "".join(parts))

        return TimestampedResult(data.decode("utf-8", errors="replace").strip())

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        language = typing.cast(str | None, kwargs.get("language"))

        if self._normalize_waveform:
            mean = waveforms.mean(axis=-1, keepdims=True)
            var = waveforms.var(axis=-1, keepdims=True)
            waveforms = (waveforms - mean) / np.sqrt(var + 1e-5)

        features, _ = self._preprocessor(waveforms, waveforms_len)

        for i, waveform_len in enumerate(waveforms_len):
            feature_len = min(int(-(-int(waveform_len) // self._hop_length)), self._max_frames)
            audio_embeds = self._encode(features[i], max(feature_len, 1))
            yield self._decode_tokens(self._generate(audio_embeds, language))
