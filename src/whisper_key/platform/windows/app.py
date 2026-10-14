import ctypes
import msvcrt
import os


def setup():
    pass

def has_interactive_terminal():
    if os.environ.get("WHISPER_KEY_NO_TERMINAL"):
        return False
    return bool(ctypes.windll.kernel32.GetConsoleWindow())

def run_event_loop(shutdown_event):
    while not shutdown_event.wait(timeout=0.1):
        pass

def getch():
    return msvcrt.getwch()
