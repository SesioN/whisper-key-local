import ctypes
import os
import subprocess
import sys

SYNCHRONIZE = 0x00100000
WAIT_FOR_EXIT_MS = 30000
WAIT_TIMEOUT = 0x102
MB_ICONWARNING = 0x30


def wait_for_process_exit(process_id: int) -> bool:
    kernel32 = ctypes.windll.kernel32
    process_handle = kernel32.OpenProcess(SYNCHRONIZE, False, process_id)
    if not process_handle:
        return True
    wait_result = kernel32.WaitForSingleObject(process_handle, WAIT_FOR_EXIT_MS)
    kernel32.CloseHandle(process_handle)
    return wait_result != WAIT_TIMEOUT


def main():
    process_id = int(sys.argv[1])
    command = sys.argv[2:]
    if not wait_for_process_exit(process_id):
        ctypes.windll.user32.MessageBoxW(None, "Whisper Key did not close in time, so it was not restarted. Start it again manually.", "Whisper Key", MB_ICONWARNING)
        return
    console_flag = subprocess.DETACHED_PROCESS if os.environ.get("PYAPP") else subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen(command, creationflags=console_flag)


if __name__ == "__main__":
    main()
