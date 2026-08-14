"""一つのsubprocessだけの終了statusとCPU時間を回収する。"""

import os
import subprocess
import time
from typing import TypeVar

ProcessOutput = TypeVar("ProcessOutput", str, bytes)


def wait_for_process(
    process: subprocess.Popen[ProcessOutput],
    *,
    timeout_seconds: float | None = None,
) -> tuple[int, float]:
    """指定processをwait4で回収し、そのprocess固有のCPU時間を返す。"""
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    while True:
        try:
            pid, status, usage = os.wait4(
                process.pid,
                0 if deadline is None else os.WNOHANG,
            )
        except InterruptedError:
            continue
        if pid != 0:
            break
        if deadline is None:
            continue
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            assert timeout_seconds is not None
            raise subprocess.TimeoutExpired(process.args, timeout_seconds)
        time.sleep(min(0.05, remaining_seconds))
    return_code = os.waitstatus_to_exitcode(status)
    process.returncode = return_code
    return return_code, usage.ru_utime + usage.ru_stime
