# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import random

from typing import List, Tuple
from itertools import combinations
from argparse import ArgumentParser
from collections import defaultdict


def generate_configurations(num_objects: int, M_range: Tuple[int, int] = (3, 5), snapshots_per_config: int = 50, target_objects_count: int = 1000) -> Tuple[List[List[int]], dict, dict]:
    pair_counts = defaultdict(int)
    object_counts = defaultdict(int)

    # Generate a list of object IDs
    objects_ids = list(range(1, num_objects + 1))
    
    # Calculate the number of configurations needed
    avg_objects_per_configuration = (M_range[0] + M_range[1]) // 2
    num_configurations = (num_objects * target_objects_count) // (snapshots_per_config * avg_objects_per_configuration)

    configurations = []
    while len(configurations) < num_configurations:
        M = min(random.randint(*M_range), len(objects_ids))

        candidates_configurations = list(combinations(objects_ids, M))
        candidates_configurations = random.sample(candidates_configurations, k=min(len(candidates_configurations), 100))

        def score(combo):
            return sum(pair_counts[tuple(sorted(p))] for p in combinations(combo, 2))
        best_configuration = min(candidates_configurations, key=score)

        for obj in best_configuration:
            object_counts[obj] += snapshots_per_config
        for pair in combinations(best_configuration, 2):
            pair_counts[tuple(sorted(pair))] += 1

        configurations.append(best_configuration)

    return configurations, object_counts, pair_counts


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate configurations for a dataset.")
    parser.add_argument("--num_objects", '-N', type=int, required=True, help="Number of objects in the dataset")
    parser.add_argument("--M_range", '-M', type=int, nargs=2, default=(3, 5), help="Range of M (min, max) for configurations")
    parser.add_argument("--snapshots_per_config", '-S', type=int, default=50, help="Number of snapshots per configuration")
    parser.add_argument("--target_objects_count", '-T', type=int, default=1000, help="Target number of objects to be counted in the dataset")
    args = parser.parse_args()

    assert args.num_objects > 0, "Number of objects must be greater than 0."

    configurations, object_counts, pair_counts = generate_configurations(
        num_objects=args.num_objects,
        M_range=args.M_range,
        snapshots_per_config=args.snapshots_per_config,
        target_objects_count=args.target_objects_count
    )

    out_file = f"dataset_configurations_N{args.num_objects}_S{args.snapshots_per_config}_T{args.target_objects_count}.txt"
    print(f"Total configurations generated: {len(configurations)}")
    print(f"Configurations saved to {out_file}")

    with open(out_file, "w") as f:
        f.write("Generated Configurations:\n")
        for config in configurations:
            f.write(f"{config}\n")

        f.write("\nObject Counts:\n")
        object_counts = dict(sorted(object_counts.items(), key=lambda x: x[0]))
        for obj, count in object_counts.items():
            f.write(f"Object {obj}: {count} appearances\n")

        f.write("\nPair Counts:\n")
        for pair, count in pair_counts.items():
            f.write(f"Pair {pair}: {count} appearances\n")
