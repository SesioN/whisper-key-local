import ctypes
import os
import subprocess
import sys

SYNCHRONIZE = 0x00100000
WAIT_FOR_EXIT_MS = 30000


def wait_for_process_exit(process_id: int):
    kernel32 = ctypes.windll.kernel32
    process_handle = kernel32.OpenProcess(SYNCHRONIZE, False, process_id)
    if process_handle:
        kernel32.WaitForSingleObject(process_handle, WAIT_FOR_EXIT_MS)
        kernel32.CloseHandle(process_handle)


def main():
    process_id = int(sys.argv[1])
    command = sys.argv[2:]
    wait_for_process_exit(process_id)
    console_flag = subprocess.DETACHED_PROCESS if os.environ.get("PYAPP") else subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen(command, creationflags=console_flag)


if __name__ == "__main__":
    main()
