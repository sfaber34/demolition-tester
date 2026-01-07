# demolition-tester

Tiny Python 3 project to hit the deterministic derby endpoint and verify it returns **identical results** for the same integer seed across repeated runs.

Endpoint:
- `http://localhost:3000/api/derby/outcome?seed=<int>`

## Terminal usage (python3)

### Available commands

- `consistency <n_runs> <seed>`: run the same seed N times and verify results match
- `gametime <n_runs>`: run N games (random seeds) and print game time percentiles
- `winrate <n_runs>`: run N games (random seeds) and print winner percentages (including draws)

### `consistency`

Run the consistency test (**N runs** of the **same seed**):

```bash
python3 -m demolition_tester consistency <n_runs> <seed>
python3 -m demolition_tester consistency 50 123
```

While running, it prints a percent-complete progress line to stderr.
If any run returns `"completed": false` and `"phase" != "gameover"`, the tools also report a `bad_state` count and the associated seed(s).

### `gametime`

Run game time statistics (**N runs** with **random seeds**):

```bash
python3 -m demolition_tester gametime <n_runs>
python3 -m demolition_tester gametime 200
```

It prints: min, p1, p10, p25, p50, p75, p90, p99, max (all in seconds).

### `winrate`

Run winner breakdown (**N runs** with **random seeds**; includes draws when `"winner": null`):

```bash
python3 -m demolition_tester winrate <n_runs>
python3 -m demolition_tester winrate 1000
```

It prints in this order: `car-1`, `car-2`, `car-3`, `car-4`, `draw`.

Change the port (pick one):

```bash
# Option A: CLI flag
python3 -m demolition_tester consistency 50 123 --port 4000

# Option B: env var
DERBY_PORT=4000 python3 -m demolition_tester consistency 50 123

# Option C: full base URL
DERBY_BASE_URL=http://localhost:4000 python3 -m demolition_tester consistency 50 123
```

Machine-readable output:

```bash
python3 -m demolition_tester consistency 50 123 --json
```

If games are slow on your machine, increase the timeout:

```bash
python3 -m demolition_tester consistency 50 123 --timeout 120
```

## Python usage

```python
from demolition_tester.derby import test_consistency

ok, summary = test_consistency(123, 50, port=3000)
print(ok, summary)
```
