# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

# GUI functions must be imported before any 'cv2' import to avoid conflicts with PyQt5
from dense_objects_nets.utils.gui_utils import select_runs_folders, select_files

import os
import gc
import time
import argparse
import torch

from copy import deepcopy
from dense_objects_nets.utils import io_utils, evaluation_utils


def model_stats(model):
    params = sum(p.numel() for p in model.parameters())
    model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    return params, model_mb


def evaluate_batch(evaluator, model, samples):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start.record()

    with torch.inference_mode():
        results = evaluator.evaluate(model, deepcopy(samples))

    end.record()
    torch.cuda.synchronize()

    return results, {
        "cuda_ms": start.elapsed_time(end),
        "wall_ms": (time.perf_counter() - wall_start) * 1000,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", "-ns", type=int, default=400)
    parser.add_argument("--batch_size", "-bs", type=int, default=200)
    args = parser.parse_args()

    run_folders, _ = select_runs_folders(
        base_dir="${WS_DIR}/runs",
        query_file="evaluation_config.yml",
    )

    dataset_files, dataset_names = select_files(
        base_dir="${WS_DIR}/config/dataset",
        query_file="**.yml",
        title="Select Evaluation Dataset",
    )

    training_files, _ = select_files(
        base_dir="${WS_DIR}/config/training",
        query_file="**.yml",
        title="Select Training Config",
    )

    if not run_folders or len(dataset_files) != 1 or len(training_files) != 1:
        raise RuntimeError("Select one dataset, one training config, and at least one run.")

    dataset_scenes = io_utils.load_config_file(dataset_files[0])
    training_config = io_utils.load_config_file(training_files[0])
    
    samples = None
    for run_folder in run_folders:
        evaluation_config = io_utils.load_config_file(
            os.path.join(run_folder, "evaluation_config.yml")
        )

        if samples is None:
            dataset = evaluation_utils.load_class(evaluation_config["dataset_class"])(
                scenes_config=dataset_scenes,
                dataset_config=training_config["dataset"],
            )
            dataset.seed = 42
            samples = dataset.load_evaluation_samples(args.num_samples)

        if len(samples) < args.num_samples:
            raise RuntimeError(f"Requested {args.num_samples} samples, got {len(samples)}.")

        evaluator = evaluation_utils.load_class(evaluation_config["evaluator_class"])(training_config["evaluator"])

        model, _ = evaluation_utils.load_models(evaluation_config)
        model = model.eval().cuda()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        params, model_mb = model_stats(model)

        print()
        print(f"Run: {run_folder}")
        print(f"Model: {evaluation_config['model_best']}")
        print(f"Parameters: {params / 1e6:.2f} M")
        print(f"Model memory: {model_mb:.2f} MB")

        # Warmup using one full batch.
        print("Warmup...")
        with torch.inference_mode():
            evaluator.evaluate(model, deepcopy(samples[:args.batch_size]))
        torch.cuda.synchronize()

        profiles = []

        for i in range(0, args.num_samples, args.batch_size):
            batch = samples[i:i + args.batch_size]

            _, profile = evaluate_batch(
                evaluator,
                model,
                batch,
            )
            profiles.append(profile)

            print(
                f"Batch {i // args.batch_size + 1}: "
                f"CUDA={profile['cuda_ms']:.2f} ms, "
                f"wall={profile['wall_ms']:.2f} ms, "
                f"peak allocated={profile['peak_allocated_mb']:.2f} MB, "
                f"peak reserved={profile['peak_reserved_mb']:.2f} MB"
            )

        total_cuda_ms = sum(p["cuda_ms"] for p in profiles)
        total_wall_ms = sum(p["wall_ms"] for p in profiles)

        print()
        print(f"Samples: {args.num_samples}")
        print(f"CUDA time: {total_cuda_ms:.2f} ms")
        print(f"Wall time: {total_wall_ms:.2f} ms")
        print(f"CUDA time/sample: {total_cuda_ms / args.num_samples:.3f} ms")
        print(f"Wall time/sample: {total_wall_ms / args.num_samples:.3f} ms")
        print(f"Throughput: {args.num_samples / (total_wall_ms / 1000):.2f} samples/s")
        print(f"Peak allocated: {max(p['peak_allocated_mb'] for p in profiles):.2f} MB")
        print(f"Peak reserved: {max(p['peak_reserved_mb'] for p in profiles):.2f} MB")

        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()