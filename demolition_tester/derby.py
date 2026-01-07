from __future__ import annotations

import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


DEFAULT_HOST = "localhost"
DEFAULT_PORT = 3000
DEFAULT_SCHEME = "http"
DEFAULT_TIMEOUT_S = 60.0


def _env_int(name: str) -> Optional[int]:
    val = os.getenv(name)
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be an int; got {val!r}")


def build_base_url(
    *,
    host: str = DEFAULT_HOST,
    port: Optional[int] = None,
    scheme: str = DEFAULT_SCHEME,
) -> str:
    """
    Build base URL like 'http://localhost:3000'.

    Port priority:
    - explicit `port` argument
    - env var DERBY_PORT
    - DEFAULT_PORT
    """
    if port is None:
        port = _env_int("DERBY_PORT") or DEFAULT_PORT
    return f"{scheme}://{host}:{port}"


def _canonical_json(obj: Any) -> str:
    """
    Deterministic string representation for deep equality checks.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _is_bad_state(outcome: Dict[str, Any]) -> bool:
    """
    Bad state definition (per requirements):
    - "completed": false
    - and "phase" != "gameover"
    """
    completed = outcome.get("completed")
    phase = outcome.get("phase")
    return completed is False and phase != "gameover"


def _bad_state_summary(count: int, seeds: Set[int]) -> Dict[str, Any]:
    return {"count": int(count), "seeds": sorted(seeds)}


def _percentile_linear(sorted_vals: List[float], p: float) -> float:
    """
    Linear-interpolated percentile (similar to numpy default).
    `sorted_vals` must be non-empty and sorted ascending.
    """
    if not sorted_vals:
        raise ValueError("sorted_vals must be non-empty")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = (p / 100.0) * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _rand_seed_int() -> int:
    # Use a wide seed space to avoid collisions at high n_runs.
    # 53 bits keeps the value within JavaScript "safe integer" range if the backend parses as Number.
    return secrets.randbits(53)


@dataclass(frozen=True)
class DerbyClient:
    """
    Simple client for the deterministic derby outcome endpoint.
    """

    base_url: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    def outcome(self, seed: int) -> Dict[str, Any]:
        return fetch_outcome(seed, base_url=self.base_url, timeout_s=self.timeout_s)


def fetch_outcome(
    seed: int,
    *,
    base_url: Optional[str] = None,
    host: str = DEFAULT_HOST,
    port: Optional[int] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Dict[str, Any]:
    """
    Call:
      GET {base_url}/api/derby/outcome?seed=<seed>

    Provide either:
    - base_url, or
    - host/port (port can also come from env var DERBY_PORT)
    """
    if base_url is None:
        base_url = build_base_url(host=host, port=port)

    query = urllib.parse.urlencode({"seed": str(int(seed))})
    url = f"{base_url.rstrip('/')}/api/derby/outcome?{query}"

    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} from {url}. Body: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach {url}: {e}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from {url}: {e}. Raw: {raw[:500]!r}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object from {url}, got {type(data).__name__}")
    return data


def test_consistency(
    seed: int,
    n_runs: int,
    *,
    base_url: Optional[str] = None,
    host: str = DEFAULT_HOST,
    port: Optional[int] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    sleep_s: float = 0.0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Runs the same seed N times and checks if results always match (deep equality).

    Returns:
      (is_consistent, summary_dict)
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0")

    if base_url is None:
        base_url = build_base_url(host=host, port=port)

    first = fetch_outcome(seed, base_url=base_url, timeout_s=timeout_s)
    first_sig = _canonical_json(first)
    if progress_cb is not None:
        progress_cb(1, n_runs)

    bad_state_count = 0
    bad_state_seeds: Set[int] = set()
    if _is_bad_state(first):
        bad_state_count += 1
        bad_state_seeds.add(int(seed))

    mismatches = 0
    mismatch_example: Optional[Dict[str, Any]] = None
    start = time.time()

    for i in range(2, n_runs + 1):
        if sleep_s > 0:
            time.sleep(sleep_s)
        cur = fetch_outcome(seed, base_url=base_url, timeout_s=timeout_s)
        cur_sig = _canonical_json(cur)
        if _is_bad_state(cur):
            bad_state_count += 1
            bad_state_seeds.add(int(seed))
        if cur_sig != first_sig:
            mismatches += 1
            if mismatch_example is None:
                mismatch_example = {
                    "run_index": i,
                    "expected_seed_hash": first.get("seed"),
                    "actual_seed_hash": cur.get("seed"),
                    "expected_canonical": first_sig,
                    "actual_canonical": cur_sig,
                }
        if progress_cb is not None:
            progress_cb(i, n_runs)

    elapsed_s = time.time() - start
    summary: Dict[str, Any] = {
        "seed": int(seed),
        "base_url": base_url,
        "n_runs": int(n_runs),
        "mismatches": mismatches,
        "elapsed_s": elapsed_s,
        "first_seed_hash": first.get("seed"),
        "bad_state": _bad_state_summary(bad_state_count, bad_state_seeds),
    }
    if mismatch_example is not None:
        summary["mismatch_example"] = mismatch_example

    return mismatches == 0, summary


def gametime_stats(
    n_runs: int,
    *,
    base_url: Optional[str] = None,
    host: str = DEFAULT_HOST,
    port: Optional[int] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    sleep_s: float = 0.0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Runs N games with random seeds and returns gameTimeMs statistics.

    Output times are in seconds (float).
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0")

    if base_url is None:
        base_url = build_base_url(host=host, port=port)

    times_ms: List[float] = []
    seeds: List[int] = []
    min_pair: Optional[Tuple[float, int]] = None  # (time_ms, seed)
    max_pair: Optional[Tuple[float, int]] = None  # (time_ms, seed)
    bad_state_count = 0
    bad_state_seeds: Set[int] = set()
    start = time.time()

    for i in range(1, n_runs + 1):
        if sleep_s > 0 and i > 1:
            time.sleep(sleep_s)

        seed = _rand_seed_int()
        data = fetch_outcome(seed, base_url=base_url, timeout_s=timeout_s)
        if _is_bad_state(data):
            bad_state_count += 1
            bad_state_seeds.add(seed)
        if "gameTimeMs" not in data:
            raise RuntimeError(f"Missing 'gameTimeMs' in response for seed={seed}. Keys: {sorted(data.keys())}")
        try:
            t_ms = float(data["gameTimeMs"])
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Invalid 'gameTimeMs'={data['gameTimeMs']!r} for seed={seed}") from e

        times_ms.append(t_ms)
        seeds.append(seed)
        if min_pair is None or t_ms < min_pair[0]:
            min_pair = (t_ms, seed)
        if max_pair is None or t_ms > max_pair[0]:
            max_pair = (t_ms, seed)

        if progress_cb is not None:
            progress_cb(i, n_runs)

    elapsed_s = time.time() - start
    times_s = sorted(t / 1000.0 for t in times_ms)
    assert min_pair is not None and max_pair is not None

    stats = {
        "min_s": min_pair[0] / 1000.0,
        "min_seed": min_pair[1],
        "p1_s": _percentile_linear(times_s, 1),
        "p10_s": _percentile_linear(times_s, 10),
        "p25_s": _percentile_linear(times_s, 25),
        "p50_s": _percentile_linear(times_s, 50),
        "p75_s": _percentile_linear(times_s, 75),
        "p90_s": _percentile_linear(times_s, 90),
        "p99_s": _percentile_linear(times_s, 99),
        "max_s": max_pair[0] / 1000.0,
        "max_seed": max_pair[1],
    }

    return {
        "base_url": base_url,
        "n_runs": int(n_runs),
        "elapsed_s": elapsed_s,
        "bad_state": _bad_state_summary(bad_state_count, bad_state_seeds),
        "stats": stats,
        # helpful for debugging/repro if anything looks odd; keep small-ish
        "example_seeds": seeds[: min(10, len(seeds))],
    }


def winrate_stats(
    n_runs: int,
    *,
    base_url: Optional[str] = None,
    host: str = DEFAULT_HOST,
    port: Optional[int] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    sleep_s: float = 0.0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Runs N games with random seeds and returns winner breakdown.

    Winner is bucketed by winner.id when present; draws are tracked when winner is null/missing.
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0")

    if base_url is None:
        base_url = build_base_url(host=host, port=port)

    counts: Counter[str] = Counter()
    bad_state_count = 0
    bad_state_seeds: Set[int] = set()
    start = time.time()

    for i in range(1, n_runs + 1):
        if sleep_s > 0 and i > 1:
            time.sleep(sleep_s)

        seed = _rand_seed_int()
        data = fetch_outcome(seed, base_url=base_url, timeout_s=timeout_s)
        if _is_bad_state(data):
            bad_state_count += 1
            bad_state_seeds.add(seed)
        winner = data.get("winner", None)
        if winner is None:
            counts["draw"] += 1
        elif isinstance(winner, dict):
            wid = winner.get("id") or winner.get("name") or "unknown-winner"
            counts[str(wid)] += 1
        else:
            counts["unknown-winner"] += 1

        if progress_cb is not None:
            progress_cb(i, n_runs)

    elapsed_s = time.time() - start

    ordered_keys = ["car-1", "car-2", "car-3", "car-4", "draw"]
    breakdown = []
    for k in ordered_keys:
        c = counts.get(k, 0)
        breakdown.append({"winner": k, "count": int(c), "pct": (c / n_runs) * 100.0})

    # Preserve any unexpected winners (so they don't get silently dropped).
    other_breakdown = []
    for k, c in counts.items():
        if k in ordered_keys:
            continue
        other_breakdown.append({"winner": k, "count": int(c), "pct": (c / n_runs) * 100.0})
    other_breakdown.sort(key=lambda r: (-r["pct"], r["winner"]))

    return {
        "base_url": base_url,
        "n_runs": int(n_runs),
        "elapsed_s": elapsed_s,
        "bad_state": _bad_state_summary(bad_state_count, bad_state_seeds),
        "breakdown": breakdown,
        "other_breakdown": other_breakdown,
    }


