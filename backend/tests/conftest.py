from __future__ import annotations

import os
import sys


_EXIT_STATUS = 0


def pytest_sessionfinish(session, exitstatus: int) -> None:
    global _EXIT_STATUS
    _EXIT_STATUS = exitstatus


def pytest_unconfigure(config) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS)
