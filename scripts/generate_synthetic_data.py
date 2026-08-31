"""CLI entry point for regenerating seed data on demand.

The actual generator lives in src/recommender/utils/synthetic_data.py
so DataIngestion can call it directly when no seed data is found. This
script is a thin wrapper for manually regenerating it - e.g. after
editing the SKILLS or CAREER_GOALS taxonomy - without deleting
data/seed and re-running the whole training pipeline.

Run: python scripts/generate_synthetic_data.py [--users 300] [--seed 42]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender.utils.synthetic_data import generate_seed_data  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/seed")
    args = parser.parse_args()

    written = generate_seed_data(args.output_dir, num_users=args.users, seed=args.seed)
    for name, path in written.items():
        print(f"{name}: {path}")
