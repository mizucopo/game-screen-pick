"""一つのsubprocessだけの終了statusとCPU時間を回収する。"""

import os
import subprocess
from typing import TypeVar

ProcessOutput = TypeVar("ProcessOutput", str, bytes)


def wait_for_process(
    process: subprocess.Popen[ProcessOutput],
) -> tuple[int, float]:
    """指定processをwait4で回収し、そのprocess固有のCPU時間を返す。"""
    while True:
        try:
            _pid, status, usage = os.wait4(process.pid, 0)
            break
        except InterruptedError:
            continue
    return_code = os.waitstatus_to_exitcode(status)
    process.returncode = return_code
    return return_code, usage.ru_utime + usage.ru_stime
