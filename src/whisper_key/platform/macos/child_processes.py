import atexit
import os
import subprocess
import weakref

_children = weakref.WeakSet()

_WATCHDOG_SCRIPT = 'while kill -0 "$1" 2>/dev/null && kill -0 "$2" 2>/dev/null; do sleep 1; done; kill -9 "$2" 2>/dev/null'


def _kill_children():
    for process in list(_children):
        if process.poll() is None:
            process.kill()


atexit.register(_kill_children)


def tie_to_current_process(process):
    _children.add(process)
    subprocess.Popen(["/bin/sh", "-c", _WATCHDOG_SCRIPT, "watchdog", str(os.getpid()), str(process.pid)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
