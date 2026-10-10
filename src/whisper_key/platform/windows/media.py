import asyncio

import comtypes
from pycaw.pycaw import AudioUtilities
from winrt.windows.media.control import (
    GlobalSystemMediaTransportControlsSessionManager as SessionManager,
    GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus,
)


def init_thread():
    comtypes.CoInitialize()


def _endpoint_volume():
    return AudioUtilities.GetSpeakers().EndpointVolume


def is_output_muted() -> bool:
    return bool(_endpoint_volume().GetMute())


def set_output_muted(muted: bool):
    _endpoint_volume().SetMute(int(muted), None)


async def _pause_playing_sessions() -> list:
    manager = await SessionManager.request_async()
    paused_app_ids = []
    for session in manager.get_sessions():
        if session.get_playback_info().playback_status != PlaybackStatus.PLAYING:
            continue
        if await session.try_pause_async():
            paused_app_ids.append(session.source_app_user_model_id)
    return paused_app_ids


async def _resume_sessions(app_ids: list):
    manager = await SessionManager.request_async()
    for session in manager.get_sessions():
        if session.source_app_user_model_id in app_ids:
            await session.try_play_async()


def pause_playing_media() -> list:
    return asyncio.run(_pause_playing_sessions())


def resume_media(paused_media: list):
    asyncio.run(_resume_sessions(paused_media))
