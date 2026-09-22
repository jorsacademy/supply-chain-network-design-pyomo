import json
import subprocess
import sys

from bilevel_pricing import run_benchmark


def test_benchmark_has_zero_reformulation_gap() -> None:
    result = run_benchmark()
    assert result.absolute_profit_gap <= 1e-6
    assert 0.0 <= result.channel_efficiency <= 1.0 + 1e-8
    assert result.centralized_channel_profit + 1e-8 >= result.channel_profit


def test_cli_outputs_json() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "bilevel_pricing"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["absolute_profit_gap"] <= 1e-6
    assert len(payload["wholesale_price"]) == 3
    assert len(payload["quantity"]) == 3
