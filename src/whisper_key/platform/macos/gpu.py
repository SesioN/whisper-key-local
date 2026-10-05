def detect_ct2_variant() -> str:
    return 'cpu'


def detect_gpu_class() -> tuple:
    return None, None


def has_vulkan_driver() -> bool:
    return False


def detect_and_print(configured_device):
    return (None, None, False)
