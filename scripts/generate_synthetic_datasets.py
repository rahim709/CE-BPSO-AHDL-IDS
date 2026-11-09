"""Generate explicitly labelled schema-inspired fixtures for missing datasets."""
from __future__ import annotations
import argparse
import json
from src.data.synthetic import generate_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='datasets/synthetic')
    parser.add_argument('--rows', type=int, default=3000, help='Total rows per synthetic dataset')
    parser.add_argument('--seed', type=int, default=381)
    args = parser.parse_args()
    print(json.dumps(generate_datasets(args.output, args.rows, args.seed), indent=2))


if __name__ == '__main__':
    main()
