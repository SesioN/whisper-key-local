import ctypes
import msvcrt


def setup():
    pass

def has_interactive_terminal():
    return bool(ctypes.windll.kernel32.GetConsoleWindow())

def run_event_loop(shutdown_event):
    while not shutdown_event.wait(timeout=0.1):
        pass

def getch():
    return msvcrt.getwch()
