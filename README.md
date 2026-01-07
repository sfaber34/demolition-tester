# demolition-tester

Tiny Python 3 project to hit the deterministic derby endpoint and verify it returns **identical results** for the same integer seed across repeated runs.

Endpoint:
- `http://localhost:3000/api/derby/outcome?seed=<int>`

## Terminal usage (python3)

Run the consistency test (**N runs** of the **same seed**):

```bash
python3 -m demolition_tester consistency <n_runs> <seed>
python3 -m demolition_tester consistency 50 123
```

While running, it prints a percent-complete progress line to stderr.

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
