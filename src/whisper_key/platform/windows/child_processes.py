import logging

import win32api
import win32con
import win32job

_job_handle = None


def _get_job():
    global _job_handle
    if _job_handle is None:
        _job_handle = win32job.CreateJobObject(None, "")
        limits = win32job.QueryInformationJobObject(_job_handle, win32job.JobObjectExtendedLimitInformation)
        limits["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(_job_handle, win32job.JobObjectExtendedLimitInformation, limits)
    return _job_handle


def tie_to_current_process(process):
    process_handle = None
    try:
        process_handle = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, process.pid)
        win32job.AssignProcessToJobObject(_get_job(), process_handle)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not tie child process {process.pid} to the app: {e}")
    finally:
        if process_handle:
            win32api.CloseHandle(process_handle)
