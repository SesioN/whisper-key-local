import logging
import os
from typing import Optional

from faster_whisper.utils import _MODELS

WHISPER_FAMILY = "whisper"
ONNX_FAMILY = "onnx_asr"


class ModelRegistry:
    DEFAULT_CACHE_PREFIX = "models--Systran--faster-whisper-"

    def __init__(self, whisper_models_config: dict = None, streaming_models_config: dict = None):
        self.whisper_models = {}
        self.streaming_models = {}
        self.logger = logging.getLogger(__name__)

        if whisper_models_config:
            for key, config in whisper_models_config.items():
                if isinstance(config, dict):
                    self.whisper_models[key] = ModelDefinition(key, config, model_type="whisper")

        if streaming_models_config:
            for key, config in streaming_models_config.items():
                if isinstance(config, dict):
                    self.streaming_models[key] = ModelDefinition(key, config, model_type="streaming")

    def get_model(self, key: str):
        return self.whisper_models.get(key)

    def get_source(self, key: str) -> str:
        model = self.get_model(key)
        return model.source if model else key

    def get_load_source(self, key: str) -> str:
        if self.is_whisper_model_downloaded(key):
            return self.get_whisper_model_dir(key)
        return self.get_source(key)

    def get_hf_repo(self, key: str) -> Optional[str]:
        model = self.get_model(key)
        source = model.source if model else key
        if model and model.is_local_path:
            return None
        if "/" in source:
            return source
        return _MODELS.get(source)

    def get_revision(self, key: str) -> str:
        model = self.get_model(key)
        return model.revision if model else "main"

    def is_english_only(self, key: str) -> bool:
        return key.endswith(".en") or self.get_source(key).endswith(".en")

    def is_known_model(self, key: str) -> bool:
        return key in self.whisper_models or key in _MODELS

    def supports_prompt(self, key: str) -> bool:
        model = self.get_model(key)
        return model.supports_prompt if model else True

    def get_cache_folder(self, key: str) -> str:
        model = self.get_model(key)
        if not model:
            return f"{self.DEFAULT_CACHE_PREFIX}{key}"
        return model.cache_folder

    def get_models_by_group(self, group: str) -> list:
        return [m for m in self.whisper_models.values() if m.group == group and m.enabled]

    def get_groups_ordered(self) -> list:
        return ["official", "custom", "onnx"]

    def get_engine_family(self, key: str) -> str:
        model = self.get_model(key)
        return model.engine if model else WHISPER_FAMILY

    def get_onnx_model_dir(self, key: str) -> Optional[str]:
        model = self.get_model(key)
        if not model or model.engine != ONNX_FAMILY:
            return None
        from .runtime_loader import get_runtimes_dir
        return str(get_runtimes_dir().parent / "models" / model.source.replace("/", "--") / model.revision)

    def get_onnx_complete_marker(self, key: str) -> Optional[str]:
        model_dir = self.get_onnx_model_dir(key)
        if not model_dir:
            return None
        return os.path.join(model_dir, f"download-complete-{self.get_model(key).quantization or 'fp32'}.json")

    def is_onnx_model_downloaded(self, key: str) -> bool:
        marker = self.get_onnx_complete_marker(key)
        return bool(marker) and os.path.isfile(marker)

    def get_whisper_model_dir(self, key: str) -> Optional[str]:
        if self.get_engine_family(key) != WHISPER_FAMILY:
            return None
        repo = self.get_hf_repo(key)
        if not repo:
            return None
        from .runtime_loader import get_runtimes_dir
        return str(get_runtimes_dir().parent / "models" / repo.replace("/", "--") / self.get_revision(key))

    def get_whisper_complete_marker(self, key: str) -> Optional[str]:
        model_dir = self.get_whisper_model_dir(key)
        if not model_dir:
            return None
        return os.path.join(model_dir, "download-complete-ct2.json")

    def is_whisper_model_downloaded(self, key: str) -> bool:
        marker = self.get_whisper_complete_marker(key)
        return bool(marker) and os.path.isfile(marker)

    def get_hf_cache_path(self) -> str:
        userprofile = os.environ.get('USERPROFILE')
        if userprofile:
            return os.path.join(userprofile, '.cache', 'huggingface', 'hub')
        return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')

    def is_model_cached(self, key: str) -> bool:
        model = self.get_model(key)
        if model and model.engine == ONNX_FAMILY:
            return self.is_onnx_model_downloaded(key)
        if model and model.is_local_path:
            return os.path.exists(os.path.join(model.source, 'model.bin'))
        if self.is_whisper_model_downloaded(key):
            return True
        cache_folder = self.get_cache_folder(key)
        if not cache_folder:
            return False
        snapshots_dir = os.path.join(self.get_hf_cache_path(), cache_folder, 'snapshots')
        if not os.path.isdir(snapshots_dir):
            return False
        return any(os.path.isfile(os.path.join(snapshots_dir, snapshot, 'model.bin')) for snapshot in os.listdir(snapshots_dir))

    def _is_streaming_model_cached(self, key: str) -> bool:
        model = self.streaming_models.get(key)
        if not model or not model.files:
            return False

        snapshot_path = self._get_streaming_snapshot_path(key)
        if not snapshot_path:
            return False

        for file_path in model.files.values():
            if not os.path.exists(os.path.join(snapshot_path, file_path)):
                return False
        return True

    def _get_streaming_snapshot_path(self, key: str) -> Optional[str]:
        model = self.streaming_models.get(key)
        if not model:
            return None

        model_dir = os.path.join(self.get_hf_cache_path(), model.cache_folder)
        snapshots_dir = os.path.join(model_dir, 'snapshots')

        if not os.path.exists(snapshots_dir):
            return None

        snapshots = os.listdir(snapshots_dir)
        if not snapshots:
            return None

        return os.path.join(snapshots_dir, snapshots[0])

    def get_streaming_model_path(self, key: str) -> Optional[tuple]:
        model = self.streaming_models.get(key)
        if not model:
            return None

        if not self._is_streaming_model_cached(key):
            if not self.download_streaming_model(key):
                return None

        snapshot_path = self._get_streaming_snapshot_path(key)
        if not snapshot_path:
            return None

        return snapshot_path, model.files

    def download_streaming_model(self, key: str) -> bool:
        model = self.streaming_models.get(key)
        if not model:
            return False

        try:
            from huggingface_hub import snapshot_download
            self.logger.info(f"Downloading streaming model: {model.source}")
            print(f"   Downloading streaming model: {model.label}...")
            snapshot_download(repo_id=model.source)
            self.logger.info(f"Streaming model downloaded: {model.source}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to download streaming model {model.source}: {e}")
            return False


class ModelDefinition:
    def __init__(self, key: str, config: dict, model_type: str = "whisper"):
        self.key = key
        self.model_type = model_type
        self.source = config.get("source", key)
        self.label = config.get("label", key.title())
        self.group = config.get("group", "custom")
        self.enabled = config.get("enabled", True)
        self.files = config.get("files", {})
        self.engine = config.get("engine", WHISPER_FAMILY)
        self.revision = config.get("revision", "main")
        self.onnx_model_type = config.get("onnx_model_type")
        self.quantization = config.get("quantization")
        self.directml = config.get("directml", True)
        self.supports_prompt = config.get("supports_prompt", True)
        if not isinstance(self.supports_prompt, bool):
            logging.getLogger(__name__).warning(f"Model {key}: supports_prompt must be true or false, got {self.supports_prompt!r}; using true")
            self.supports_prompt = True
        self.max_chunk_seconds = config.get("max_chunk_seconds")
        if self.max_chunk_seconds is not None and (isinstance(self.max_chunk_seconds, bool) or not isinstance(self.max_chunk_seconds, (int, float)) or self.max_chunk_seconds <= 0):
            logging.getLogger(__name__).warning(f"Model {key}: max_chunk_seconds must be a positive number, got {self.max_chunk_seconds!r}; using the default")
            self.max_chunk_seconds = None
        self.is_local_path = self._check_is_local_path()
        self.cache_folder = self._derive_cache_folder()

    def _check_is_local_path(self) -> bool:
        if self.source.startswith("\\\\") or (len(self.source) > 2 and self.source[1] == ":"):
            return True
        if "/" in self.source:
            return os.path.exists(self.source)
        return False

    def _derive_cache_folder(self) -> str:
        if self.is_local_path:
            return None

        if "/" in self.source:
            return "models--" + self.source.replace("/", "--")

        if self.source in _MODELS:
            repo = _MODELS[self.source]
            return "models--" + repo.replace("/", "--")

        return f"{ModelRegistry.DEFAULT_CACHE_PREFIX}{self.source}"
