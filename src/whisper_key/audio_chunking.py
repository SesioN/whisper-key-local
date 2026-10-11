import numpy as np

SAMPLE_RATE = 16000


def split_at_quiet_points(audio: np.ndarray, max_seconds: float) -> list:
    chunks = []
    frame = SAMPLE_RATE // 50
    max_samples = int(max_seconds * SAMPLE_RATE)
    search_start = max_samples * 2 // 3
    while len(audio) > max_samples:
        search = audio[search_start:max_samples]
        frame_energy = np.square(search[:len(search) // frame * frame].reshape(-1, frame)).mean(axis=1)
        split = search_start + int(frame_energy.argmin()) * frame + frame // 2
        chunks.append(audio[:split])
        audio = audio[split:]
    chunks.append(audio)
    return chunks
