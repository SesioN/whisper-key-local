import atexit
import weakref

_children = weakref.WeakSet()


def _kill_children():
    for process in list(_children):
        if process.poll() is None:
            process.kill()


atexit.register(_kill_children)


def tie_to_current_process(process):
    _children.add(process)
