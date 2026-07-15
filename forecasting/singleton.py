"""
Single-instance guard for the long-running loops.

Repeated relaunches across sessions were spawning duplicate live_loop /
l3_puller processes that raced on the same md rows (per-process forming-bar
state + ON CONFLICT DO NOTHING => silent stalls). This makes a second copy
refuse to start while the first is alive.

Lock = a file under data/locks/<name>.lock holding the PID. On startup we
check whether that PID is still a live python process; if so, abort; if it's
stale (crash left the file), we take it over.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path

_LOCK_DIR = Path(__file__).resolve().parents[1] / "data" / "locks"


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower()
    except ImportError:
        pass
    # stdlib fallback (Windows + POSIX): signal 0 probes existence
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False
    except PermissionError:
        return True     # exists, owned by someone else


class AlreadyRunning(RuntimeError):
    pass


def acquire(name: str) -> Path:
    """Claim the named lock or raise AlreadyRunning. Auto-released at exit.

    Uses atomic O_EXCL create so two processes racing at the same instant
    cannot both win. If the lock exists but its holder is dead (crash), the
    stale file is taken over."""
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock = _LOCK_DIR / f"{name}.lock"

    def _claim():
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))

    try:
        _claim()
    except FileExistsError:
        try:
            other = int(lock.read_text().strip() or "0")
        except (ValueError, OSError):
            other = 0
        if other and other != os.getpid() and _pid_alive(other):
            raise AlreadyRunning(
                f"{name} already running as pid {other} "
                f"(lock {lock}); refusing to start a duplicate")
        # holder is dead — take over the stale lock
        try:
            lock.unlink()
        except OSError:
            pass
        _claim()

    @atexit.register
    def _release():
        try:
            if lock.exists() and lock.read_text().strip() == str(os.getpid()):
                lock.unlink()
        except OSError:
            pass

    return lock
