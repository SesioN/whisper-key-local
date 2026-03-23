import msvcrt
import sys
import os
import ctypes


def setup():
    pass

def ensure_console(force_show=False):
    """
    Ensures that stdout/stderr are valid.
    If force_show is True, allocates a new console window on Windows.
    If False, redirects to devnull if no console exists.
    """
    if force_show:
        # Allocate a new console
        kernel32 = ctypes.windll.kernel32
        # Check if we already have a console
        if kernel32.GetConsoleWindow() == 0:
            kernel32.AllocConsole()
            sys.stdout = open('CONOUT$', 'w', encoding='utf-8')
            sys.stderr = open('CONOUT$', 'w', encoding='utf-8')
    else:
        # If no console and no stdout/stderr (pythonw case), redirect to devnull
        # to prevent crashes on print()
        if sys.stdout is None:
            sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        if sys.stderr is None:
            sys.stderr = open(os.devnull, 'w', encoding='utf-8')

def run_event_loop(shutdown_event):
    while not shutdown_event.wait(timeout=0.1):
        pass

def getch():
    return msvcrt.getwch()
