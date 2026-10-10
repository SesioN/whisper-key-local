import asyncio


def init_thread():
    import comtypes
    comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)


def _endpoint_volume():
    from pycaw.pycaw import AudioUtilities
    return AudioUtilities.GetSpeakers().EndpointVolume


def is_output_muted() -> bool:
    return bool(_endpoint_volume().GetMute())


def set_output_muted(muted: bool):
    _endpoint_volume().SetMute(int(muted), None)


async def _pause_playing_sessions() -> list:
    from winrt.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as SessionManager,
        GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus,
    )
    manager = await SessionManager.request_async()
    paused_sessions = []
    for session in manager.get_sessions():
        try:
            if session.get_playback_info().playback_status != PlaybackStatus.PLAYING:
                continue
            if await session.try_pause_async():
                paused_sessions.append(session)
        except OSError:
            continue
    return paused_sessions


async def _resume_sessions(paused_sessions: list):
    for session in paused_sessions:
        try:
            await session.try_play_async()
        except OSError:
            continue


def pause_playing_media() -> list:
    return asyncio.run(_pause_playing_sessions())


def resume_media(paused_media: list):
    asyncio.run(_resume_sessions(paused_media))
