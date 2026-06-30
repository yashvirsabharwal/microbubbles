"""Resumable download of the public sample ultratrace.

Pure standard library (``urllib``) so the package keeps its minimal dependency
footprint -- no ``curl``/``requests`` required. Supports HTTP range resume so an
interrupted ~98 GB download can be continued in place.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import truststore
except ImportError:  # pragma: no cover - dependency is declared for normal installs.
    truststore = None
else:
    truststore.inject_into_ssl()

#: Sanitized neutral ultratrace (demodulated IQ + transmit delays + a
#: beamforming-only config) hosted on Cloudflare R2: ~98 GB, 223 acquisitions.
SAMPLE_URL = (
    "https://pub-9c1be6312b2441eb8732660783d9ee81.r2.dev/"
    "sanitized_neutral_ultratrace.h5"
)
SAMPLE_FILENAME = "sample_ultratrace.h5"

_CHUNK = 8 * 1024 * 1024  # 8 MiB
_SEGMENT = 256 * 1024 * 1024  # 256 MiB
_TIMEOUT = 30.0
# Cloudflare R2's public endpoint returns 403 for urllib's default
# ``Python-urllib/x.y`` agent, so send an explicit one.
_USER_AGENT = "ultratrace-ulm/0.1"


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, ValueError):
        return None


def _copy_range(src: Path, dst: Path, start: int, length: int, chunk: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    remaining = length
    with open(src, "rb") as in_fh, open(dst, "wb") as out_fh:
        in_fh.seek(start)
        while remaining:
            block = in_fh.read(min(chunk, remaining))
            if not block:
                break
            out_fh.write(block)
            remaining -= len(block)


def _seed_parts_from_output(out: Path, parts_dir: Path, total: int, parts: int, chunk: int) -> None:
    """Seed range part files from an existing contiguous partial download."""
    existing = out.stat().st_size if out.exists() else 0
    if not existing:
        return
    part_size = (total + parts - 1) // parts
    for index in range(parts):
        start = index * part_size
        if start >= existing:
            break
        end = min(start + part_size, total, existing)
        length = end - start
        if length <= 0:
            continue
        part = parts_dir / f"part_{index:03d}"
        if part.exists() and part.stat().st_size >= length:
            continue
        print(f"Seeding {part.name} from existing file ({_fmt_bytes(length)})")
        _copy_range(out, part, start, length, chunk)


def _download_range_part(
    *,
    url: str,
    part: Path,
    start: int,
    end: int,
    chunk: int,
    progress: dict[int, int],
    progress_lock: threading.Lock,
    index: int,
) -> Path:
    expected = end - start + 1
    part.parent.mkdir(parents=True, exist_ok=True)
    existing = part.stat().st_size if part.exists() else 0
    if existing > expected:
        part.unlink()
        existing = 0

    while existing < expected:
        range_start = start + existing
        headers = {
            "User-Agent": _USER_AGENT,
            "Range": f"bytes={range_start}-{end}",
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                mode = "ab" if resp.status == 206 else "wb"
                if mode == "wb":
                    existing = 0
                with open(part, mode) as fh:
                    while existing < expected:
                        block = resp.read(min(chunk, expected - existing))
                        if not block:
                            break
                        fh.write(block)
                        existing += len(block)
                        with progress_lock:
                            progress[index] = existing
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            existing = part.stat().st_size if part.exists() else 0
            with progress_lock:
                progress[index] = min(existing, expected)
            print(f"\n  segment {index:05d} stalled ({exc}); reconnecting ...", file=sys.stderr)
            time.sleep(2.0)
    return part


def _monitor_parallel(progress: dict[int, int], progress_lock: threading.Lock, total: int, done: threading.Event) -> None:
    last_bytes = 0
    last_time = time.monotonic()
    smoothed_rate: float | None = None
    while not done.wait(1.0):
        now = time.monotonic()
        with progress_lock:
            current = sum(progress.values())
        interval = max(now - last_time, 1e-6)
        rate = max(0.0, (current - last_bytes) / interval)
        if smoothed_rate is None:
            smoothed_rate = rate
        elif rate > 0:
            smoothed_rate = 0.35 * rate + 0.65 * smoothed_rate
        elif smoothed_rate > 0:
            smoothed_rate *= 0.85
        pct = 100.0 * current / total
        eta = _fmt_duration((total - current) / smoothed_rate) if smoothed_rate else "?"
        print(
            f"\r  {_fmt_bytes(current)} / {_fmt_bytes(total)} ({pct:.1f}%) "
            f"| {_fmt_bytes(smoothed_rate or 0)}/s | ETA {eta}        ",
            end="",
            file=sys.stderr,
            flush=True,
        )
        last_bytes = current
        last_time = now


def _read_benchmark_range(url: str, start: int, end: int, deadline: float, chunk: int) -> int:
    headers = {
        "User-Agent": _USER_AGENT,
        "Range": f"bytes={start}-{end}",
    }
    req = urllib.request.Request(url, headers=headers)
    total = 0
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        while time.monotonic() < deadline:
            block = resp.read(chunk)
            if not block:
                break
            total += len(block)
    return total


def _benchmark_connections(
    url: str,
    total: int,
    candidates: tuple[int, ...] = (4, 8),
    seconds: float = 6.0,
    chunk: int = _CHUNK,
) -> int:
    """Benchmark short throwaway range reads and return the fastest connection count."""
    print("Benchmarking download modes...")
    results: list[tuple[float, int]] = []
    span = min(total, 512 * 1024 * 1024)
    base = max(0, total // 2 - span // 2)
    for connections in candidates:
        connections = max(1, connections)
        per_conn = max(chunk, span // connections)
        deadline = time.monotonic() + seconds
        started = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=connections) as pool:
                futures = []
                for index in range(connections):
                    start = min(total - 1, base + index * per_conn)
                    end = min(total - 1, start + per_conn - 1)
                    futures.append(pool.submit(_read_benchmark_range, url, start, end, deadline, chunk))
                downloaded = sum(future.result() for future in as_completed(futures))
        except Exception as exc:
            print(f"  {connections} connection(s): failed ({exc})")
            continue
        elapsed = max(time.monotonic() - started, 1e-6)
        rate = downloaded / elapsed
        results.append((rate, connections))
        print(f"  {connections} connection(s): {_fmt_bytes(rate)}/s")
    if not results:
        print("Benchmark failed; falling back to single stream.")
        return 1
    best_rate, best_connections = max(results)
    print(f"Selected {best_connections} connection(s): {_fmt_bytes(best_rate)}/s")
    return best_connections


def download_sample_auto(
    url: str = SAMPLE_URL,
    output: str | Path = SAMPLE_FILENAME,
    *,
    force: bool = False,
    chunk: int = _CHUNK,
    recheck_seconds: float = 300.0,
) -> Path:
    """Adaptively benchmark available modes and download with the fastest one."""
    return download_sample_adaptive(
        url,
        output,
        force=force,
        chunk=chunk,
        recheck_seconds=recheck_seconds,
    )


def _range_segments(total: int, segment_size: int) -> list[tuple[int, int, int]]:
    segments = []
    index = 0
    for start in range(0, total, segment_size):
        end = min(start + segment_size - 1, total - 1)
        segments.append((index, start, end))
        index += 1
    return segments


def _seed_segments_from_output(
    out: Path,
    segments_dir: Path,
    segments: list[tuple[int, int, int]],
    chunk: int,
) -> None:
    existing = out.stat().st_size if out.exists() else 0
    if not existing:
        return
    for index, start, end in segments:
        if start >= existing:
            break
        length = min(end + 1, existing) - start
        if length <= 0:
            continue
        segment = segments_dir / f"segment_{index:05d}"
        if segment.exists() and segment.stat().st_size >= length:
            continue
        print(f"Seeding {segment.name} from existing file ({_fmt_bytes(length)})")
        _copy_range(out, segment, start, length, chunk)


def _segment_sizes(
    segments_dir: Path,
    segments: list[tuple[int, int, int]],
) -> dict[int, int]:
    sizes = {}
    for index, start, end in segments:
        expected = end - start + 1
        segment = segments_dir / f"segment_{index:05d}"
        sizes[index] = min(segment.stat().st_size, expected) if segment.exists() else 0
    return sizes


def _segment_complete(segments_dir: Path, index: int, start: int, end: int) -> bool:
    segment = segments_dir / f"segment_{index:05d}"
    return segment.exists() and segment.stat().st_size == end - start + 1


def download_sample_adaptive(
    url: str = SAMPLE_URL,
    output: str | Path = SAMPLE_FILENAME,
    *,
    force: bool = False,
    chunk: int = _CHUNK,
    segment_size: int = _SEGMENT,
    recheck_seconds: float = 300.0,
) -> Path:
    """Download with periodic benchmarking and adaptive range concurrency."""
    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    total = _remote_size(url)
    if total is None:
        print("Could not determine remote size; using single stream.")
        return download_sample(url, out, force=force, chunk=chunk)
    if force:
        out.unlink(missing_ok=True)
        shutil.rmtree(out.with_suffix(out.suffix + ".segments"), ignore_errors=True)
    if out.exists() and not force and out.stat().st_size == total:
        print(f"Already complete: {out} ({_fmt_bytes(total)})")
        return out

    segments_dir = out.with_suffix(out.suffix + ".segments")
    segments_dir.mkdir(parents=True, exist_ok=True)
    segments = _range_segments(total, segment_size)
    _seed_segments_from_output(out, segments_dir, segments, chunk)

    progress = _segment_sizes(segments_dir, segments)
    progress_lock = threading.Lock()
    done = threading.Event()
    monitor = threading.Thread(target=_monitor_parallel, args=(progress, progress_lock, total, done), daemon=True)
    monitor.start()

    selected = 1
    next_recheck = 0.0
    try:
        while True:
            incomplete = [
                (index, start, end)
                for index, start, end in segments
                if not _segment_complete(segments_dir, index, start, end)
            ]
            if not incomplete:
                break

            now = time.monotonic()
            if now >= next_recheck:
                done.set()
                monitor.join(timeout=1.0)
                print("", file=sys.stderr)
                selected = _benchmark_connections(url, total, chunk=chunk)
                print(f"Using {selected} connection(s) until next recheck.")
                done = threading.Event()
                monitor = threading.Thread(
                    target=_monitor_parallel,
                    args=(progress, progress_lock, total, done),
                    daemon=True,
                )
                monitor.start()
                next_recheck = time.monotonic() + recheck_seconds

            batch = incomplete[:selected]
            with ThreadPoolExecutor(max_workers=selected) as pool:
                futures = [
                    pool.submit(
                        _download_range_part,
                        url=url,
                        part=segments_dir / f"segment_{index:05d}",
                        start=start,
                        end=end,
                        chunk=chunk,
                        progress=progress,
                        progress_lock=progress_lock,
                        index=index,
                    )
                    for index, start, end in batch
                ]
                for future in as_completed(futures):
                    future.result()
    finally:
        done.set()
        monitor.join(timeout=1.0)
        print("", file=sys.stderr)

    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as out_fh:
        for index, start, end in segments:
            segment = segments_dir / f"segment_{index:05d}"
            expected = end - start + 1
            actual = segment.stat().st_size
            if actual != expected:
                raise RuntimeError(
                    f"{segment} is incomplete: {_fmt_bytes(actual)} / {_fmt_bytes(expected)}"
                )
            with open(segment, "rb") as in_fh:
                while True:
                    block = in_fh.read(chunk)
                    if not block:
                        break
                    out_fh.write(block)
    if tmp.stat().st_size != total:
        raise RuntimeError(f"Assembled file has wrong size: {_fmt_bytes(tmp.stat().st_size)} / {_fmt_bytes(total)}")
    tmp.replace(out)
    shutil.rmtree(segments_dir)
    print(f"Saved {out} ({_fmt_bytes(total)})")
    return out


def download_sample_parallel(
    url: str = SAMPLE_URL,
    output: str | Path = SAMPLE_FILENAME,
    *,
    force: bool = False,
    chunk: int = _CHUNK,
    connections: int = 8,
) -> Path:
    """Download ``url`` using parallel HTTP range requests."""
    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    total = _remote_size(url)
    if total is None:
        raise RuntimeError("Could not determine remote size for parallel download")
    if force and out.exists():
        out.unlink()
    if out.exists() and out.stat().st_size == total:
        print(f"Already complete: {out} ({_fmt_bytes(total)})")
        return out

    connections = max(1, int(connections))
    parts_dir = out.with_suffix(out.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    _seed_parts_from_output(out, parts_dir, total, connections, chunk)

    part_size = (total + connections - 1) // connections
    progress: dict[int, int] = {}
    progress_lock = threading.Lock()
    ranges: list[tuple[int, int, int, Path]] = []
    for index in range(connections):
        start = index * part_size
        if start >= total:
            break
        end = min(start + part_size - 1, total - 1)
        part = parts_dir / f"part_{index:03d}"
        existing = part.stat().st_size if part.exists() else 0
        progress[index] = min(existing, end - start + 1)
        ranges.append((index, start, end, part))

    done = threading.Event()
    monitor = threading.Thread(target=_monitor_parallel, args=(progress, progress_lock, total, done), daemon=True)
    monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures = [
                pool.submit(
                    _download_range_part,
                    url=url,
                    part=part,
                    start=start,
                    end=end,
                    chunk=chunk,
                    progress=progress,
                    progress_lock=progress_lock,
                    index=index,
                )
                for index, start, end, part in ranges
            ]
            for future in as_completed(futures):
                future.result()
    finally:
        done.set()
        monitor.join(timeout=1.0)
        print("", file=sys.stderr)

    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as out_fh:
        for _, start, end, part in ranges:
            expected = end - start + 1
            actual = part.stat().st_size
            if actual != expected:
                raise RuntimeError(f"{part} is incomplete: {_fmt_bytes(actual)} / {_fmt_bytes(expected)}")
            with open(part, "rb") as in_fh:
                while True:
                    block = in_fh.read(chunk)
                    if not block:
                        break
                    out_fh.write(block)
    if tmp.stat().st_size != total:
        raise RuntimeError(f"Assembled file has wrong size: {_fmt_bytes(tmp.stat().st_size)} / {_fmt_bytes(total)}")
    tmp.replace(out)
    shutil.rmtree(parts_dir)
    print(f"Saved {out} ({_fmt_bytes(total)})")
    return out


def download_sample(
    url: str = SAMPLE_URL,
    output: str | Path = SAMPLE_FILENAME,
    *,
    force: bool = False,
    chunk: int = _CHUNK,
    max_attempts: int = 100,
) -> Path:
    """Download ``url`` to ``output``, resuming a partial file when possible.

    Returns the resolved output path. Re-running after a complete download is a
    no-op unless ``force`` is set.
    """
    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    total = _remote_size(url)
    if force and out.exists():
        out.unlink()

    existing = out.stat().st_size if out.exists() else 0
    if total is not None and existing == total:
        print(f"Already complete: {out} ({_fmt_bytes(total)})")
        return out
    if total is not None and existing > total:
        # Local file is larger than remote -- assume stale, restart.
        out.unlink()
        existing = 0

    attempts = 0
    last_size = -1
    while True:
        existing = out.stat().st_size if out.exists() else 0
        if total is not None and existing == total:
            print(f"Saved {out} ({_fmt_bytes(existing)})")
            return out
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Download incomplete after {max_attempts} attempts: "
                f"{_fmt_bytes(existing)}"
                + (f" / {_fmt_bytes(total)}" if total is not None else "")
            )
        if existing == last_size:
            raise RuntimeError(f"Download made no progress at {_fmt_bytes(existing)}")
        last_size = existing
        attempts += 1

        headers = {"User-Agent": _USER_AGENT}
        if existing:
            headers["Range"] = f"bytes={existing}-"
            print(f"Resuming from {_fmt_bytes(existing)} ...")

        req = urllib.request.Request(url, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=_TIMEOUT)
        except urllib.error.HTTPError as exc:
            if exc.code == 416:  # Range Not Satisfiable -> already have it all.
                print(f"Already complete: {out} ({_fmt_bytes(existing)})")
                return out
            raise

        with resp:
            # Server honored the range request -> append; otherwise restart.
            if existing and resp.status == 206:
                mode = "ab"
                done = existing
            else:
                mode = "wb"
                done = 0
            grand_total = total
            cl = resp.headers.get("Content-Length")
            if grand_total is None and cl is not None:
                grand_total = done + int(cl)

            start_done = done
            start_time = time.monotonic()
            last_print = 0.0
            with open(out, mode) as fh:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    now = time.monotonic()
                    if now - last_print < 0.5:
                        continue
                    last_print = now
                    elapsed = max(now - start_time, 1e-6)
                    rate = (done - start_done) / elapsed
                    rate_text = f"{_fmt_bytes(rate)}/s"
                    if grand_total and rate > 0:
                        pct = 100.0 * done / grand_total
                        eta = _fmt_duration((grand_total - done) / rate)
                        bar = (
                            f"{_fmt_bytes(done)} / {_fmt_bytes(grand_total)} "
                            f"({pct:.1f}%) | {rate_text} | ETA {eta}"
                        )
                    else:
                        bar = f"{_fmt_bytes(done)} | {rate_text}"
                    print(f"\r  {bar}        ", end="", file=sys.stderr, flush=True)
        print("", file=sys.stderr)
        if total is None:
            print(f"Saved {out} ({_fmt_bytes(out.stat().st_size)})")
            return out
        print(f"Connection ended at {_fmt_bytes(out.stat().st_size)}; resuming ...")
