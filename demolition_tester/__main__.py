from __future__ import annotations

import argparse
import json
import os
import sys

from .derby import DEFAULT_TIMEOUT_S, build_base_url, test_consistency


def _env_default(name: str, fallback: str | None = None) -> str | None:
    val = os.getenv(name)
    if val is None or val == "":
        return fallback
    return val


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demolition_tester")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cons = sub.add_parser("consistency", help="Run the same seed N times and assert results match")
    p_cons.add_argument("n_runs", type=int, help="Number of runs")
    p_cons.add_argument("seed", type=int, help="Integer seed to pass to the endpoint")
    p_cons.add_argument("--base-url", default=_env_default("DERBY_BASE_URL"), help="Override full base URL")
    p_cons.add_argument("--host", default=_env_default("DERBY_HOST", "localhost"), help="Host (default: localhost)")
    p_cons.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port (default: env DERBY_PORT or 3000)",
    )
    p_cons.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"HTTP timeout seconds (default: {DEFAULT_TIMEOUT_S:g})",
    )
    p_cons.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between runs (default: 0)")
    p_cons.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Print JSON summary (machine readable)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "consistency":
        base_url = args.base_url or build_base_url(host=args.host, port=args.port)

        def progress_cb(done: int, total: int) -> None:
            if total <= 0:
                return
            pct = int(round((done / total) * 100))
            msg = f"\rProgress: {done}/{total} ({pct}%)"
            print(msg, end="", file=sys.stderr, flush=True)

        ok, summary = test_consistency(
            args.seed,
            args.n_runs,
            base_url=base_url,
            timeout_s=args.timeout,
            sleep_s=args.sleep,
            progress_cb=None if args.as_json else progress_cb,
        )
        if not args.as_json:
            print("", file=sys.stderr)  # newline after progress line

        if args.as_json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            status = "CONSISTENT" if ok else "INCONSISTENT"
            print(f"{status}: seed={summary['seed']} runs={summary['n_runs']} mismatches={summary['mismatches']}")
            print(f"base_url={summary['base_url']}")
            if "first_seed_hash" in summary and summary["first_seed_hash"] is not None:
                print(f"first_seed_hash={summary['first_seed_hash']}")
            if "mismatch_example" in summary:
                print("mismatch_example:")
                print(json.dumps(summary["mismatch_example"], indent=2, sort_keys=True))

        return 0 if ok else 2

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


