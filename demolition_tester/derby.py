from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


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

    mismatches = 0
    mismatch_example: Optional[Dict[str, Any]] = None
    start = time.time()

    for i in range(2, n_runs + 1):
        if sleep_s > 0:
            time.sleep(sleep_s)
        cur = fetch_outcome(seed, base_url=base_url, timeout_s=timeout_s)
        cur_sig = _canonical_json(cur)
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

    elapsed_s = time.time() - start
    summary: Dict[str, Any] = {
        "seed": int(seed),
        "base_url": base_url,
        "n_runs": int(n_runs),
        "mismatches": mismatches,
        "elapsed_s": elapsed_s,
        "first_seed_hash": first.get("seed"),
    }
    if mismatch_example is not None:
        summary["mismatch_example"] = mismatch_example

    return mismatches == 0, summary


