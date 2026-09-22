from __future__ import annotations

import argparse
import json

from .benchmark import run_benchmark
from .model import demo_instance, random_instance


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the optimistic bilevel pricing benchmark.")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--products", type=int, default=3)
    args = parser.parse_args()

    instance = (
        demo_instance()
        if args.random_seed is None
        else random_instance(args.products, args.random_seed)
    )
    result = run_benchmark(instance)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
