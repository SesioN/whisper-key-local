from whisper_key.runtime_options import FASTER_WHISPER, INSTALLED, Runtime, choose_compute_type


def runtime(device, compute_types):
    return Runtime("cpu" if device == "cpu" else "cuda", "label", FASTER_WHISPER, device, INSTALLED,
                   compute_types=frozenset(compute_types))


def test_requested_compute_type_kept_when_supported():
    assert choose_compute_type(runtime("cpu", {"int8", "float32"}), "float32") == "float32"


def test_unsupported_request_falls_back_to_device_preference():
    assert choose_compute_type(runtime("cuda", {"int8", "float16", "float32"}), "bfloat16") == "float16"
