# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import shutil
import random
import argparse

from tqdm import tqdm
from collections import defaultdict, Counter

from dense_objects_nets.utils import io_utils


def select_snapshots_with_global_quota(
    object_to_snapshot_keys,
    snapshot_object_counts,
    target_samples,
    rng,
):
    """
    Select snapshots globally so that the total number of instances for each
    object class is as close as possible to target_samples.

    Each source snapshot can be selected at most once.

    The algorithm repeatedly:
      1. finds an object that is still below target;
      2. prioritizes objects that are harder to satisfy with the remaining data;
      3. selects the snapshot that:
         - avoids overflow if possible;
         - minimizes unavoidable overflow;
         - fills as much global deficit as possible;
         - helps multiple underrepresented objects simultaneously.

    Returns:
        selected_by_object:
            Dict mapping the object that triggered each selection to the list
            of selected snapshot keys.

        global_counts:
            Final number of object instances across all selected snapshots.

        selected_keys:
            Set of all selected source snapshots.
    """
    obj_ids = sorted(object_to_snapshot_keys.keys())

    global_counts = defaultdict(int)
    selected_by_object = defaultdict(list)
    selected_keys = set()

    while True:
        deficits = {
            obj_id: max(0, target_samples - global_counts[obj_id])
            for obj_id in obj_ids
        }

        unmet_objects = [
            obj_id
            for obj_id in obj_ids
            if deficits[obj_id] > 0
        ]

        if not unmet_objects:
            break

        # Determine which unmet object is currently the most constrained.
        #
        # We compare the remaining instance capacity against the remaining
        # deficit. Lower capacity/deficit means the object is harder to satisfy.
        focus_candidates = {}
        constrainedness = {}

        for obj_id in unmet_objects:
            candidates = [
                key
                for key in object_to_snapshot_keys[obj_id]
                if key not in selected_keys
            ]

            focus_candidates[obj_id] = candidates

            available_capacity = sum(
                snapshot_object_counts[key].get(obj_id, 0)
                for key in candidates
            )

            if available_capacity <= 0:
                constrainedness[obj_id] = float("inf")
            else:
                constrainedness[obj_id] = (
                    available_capacity / deficits[obj_id]
                )

        feasible_objects = [
            obj_id
            for obj_id in unmet_objects
            if focus_candidates[obj_id]
        ]

        if not feasible_objects:
            break

        focus_obj = min(
            feasible_objects,
            key=lambda obj_id: (
                constrainedness[obj_id],
                len(focus_candidates[obj_id]),
                -deficits[obj_id],
                obj_id,
            ),
        )

        candidates = list(focus_candidates[focus_obj])
        rng.shuffle(candidates)

        best_key = None
        best_score = None

        for key in candidates:
            counts = snapshot_object_counts[key]

            primary_increment = counts.get(focus_obj, 0)
            if primary_increment <= 0:
                continue

            primary_deficit = deficits[focus_obj]
            primary_gain = min(primary_increment, primary_deficit)

            if primary_gain <= 0:
                continue

            useful_gain = 0
            new_overflow = 0
            total_increment = 0
            helped_objects = 0
            saturated_increment = 0

            for obj_id, increment in counts.items():
                total_increment += increment

                current_count = global_counts[obj_id]
                deficit = max(
                    0,
                    target_samples - current_count,
                )

                gain = min(increment, deficit)
                useful_gain += gain

                if gain > 0:
                    helped_objects += 1

                if current_count >= target_samples:
                    saturated_increment += increment

                overflow_before = max(
                    0,
                    current_count - target_samples,
                )

                overflow_after = max(
                    0,
                    current_count + increment - target_samples,
                )

                new_overflow += (
                    overflow_after - overflow_before
                )

            # Lexicographic score:
            #
            # 1. Prefer snapshots introducing zero overflow.
            # 2. Otherwise minimize total new overflow.
            # 3. Maximize total useful instances contributed to unmet classes.
            # 4. Prefer snapshots helping more underrepresented classes.
            # 5. Prefer greater gain for the currently constrained class.
            # 6. Avoid adding instances to already saturated classes.
            # 7. Avoid unnecessarily dense snapshots.
            score = (
                int(new_overflow == 0),
                -new_overflow,
                useful_gain,
                helped_objects,
                primary_gain,
                -saturated_increment,
                -total_increment,
            )

            if best_score is None or score > best_score:
                best_score = score
                best_key = key

        if best_key is None:
            # Cannot improve the chosen object anymore.
            #
            # Remove it from future consideration by marking all its remaining
            # candidate snapshots as unavailable for focusing would be wrong,
            # because they may still help another object. Instead, check whether
            # any unmet object can still be improved.
            any_improvable = False

            for obj_id in unmet_objects:
                if any(
                    key not in selected_keys
                    and snapshot_object_counts[key].get(obj_id, 0) > 0
                    for key in object_to_snapshot_keys[obj_id]
                ):
                    any_improvable = True
                    break

            if not any_improvable:
                break

            # This case should normally not occur, but prevents an infinite loop.
            break

        selected_keys.add(best_key)
        selected_by_object[focus_obj].append(best_key)

        for obj_id, increment in snapshot_object_counts[best_key].items():
            global_counts[obj_id] += increment

    return selected_by_object, global_counts, selected_keys


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Shrink a dataset by globally selecting snapshots so that the "
            "number of instances per object class is approximately uniform."
        )
    )
    parser.add_argument("--in_path", "-in", type=str, required=True, help="Path to the original dataset root directory.")
    parser.add_argument("--target_samples", "-ts", type=int, required=True, help="Target number of instances per object class.")
    parser.add_argument("--out_prefix", "-op", type=str, default="mini", help="Output prefix for the shrunk dataset.")
    parser.add_argument("--force_single_instance", "-fsi", action="store_true", help=("Discard snapshots containing multiple instances of the same object class."),)
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for tie breaking.")
    args = parser.parse_args()

    if args.target_samples <= 0:
        raise ValueError("--target_samples must be > 0")

    rng = random.Random(args.seed)

    dataset_scenes = [
        scene
        for scene in sorted(os.listdir(args.in_path))
        if os.path.exists(
            os.path.join(args.in_path, scene, "snapshots.yml",)
        )
    ]
    if not dataset_scenes:
        raise RuntimeError(f"No scenes containing snapshots.yml found in: {args.in_path}")


    # Load the dataset snapshots
    all_snapshots = {}
    snapshot_object_counts = {}
    object_to_snapshot_keys = defaultdict(set)
    for scene in tqdm(dataset_scenes, desc="Collecting all object snapshots dicts", total=len(dataset_scenes)):
        snapshots_path = os.path.join(args.in_path, scene, "snapshots.yml")
        scene_snapshots = io_utils.load_config_file(snapshots_path)

        for snapshot_id, snapshot_dict in scene_snapshots.items():
            snapshot_objects_ids = snapshot_dict["objects_ids"]

            if args.force_single_instance:
                if len(set(snapshot_objects_ids)) < len(snapshot_objects_ids):
                    continue

            snapshot_key = (scene, snapshot_id)

            all_snapshots[snapshot_key] = snapshot_dict
            object_counts = Counter(snapshot_objects_ids)
            snapshot_object_counts[snapshot_key] = object_counts
            for obj_id in object_counts:
                object_to_snapshot_keys[obj_id].add(snapshot_key)

    if not snapshot_object_counts:
        raise RuntimeError("No valid snapshots found after filtering.")

    all_obj_ids = sorted(object_to_snapshot_keys.keys())

    print(
        f"Found {len(snapshot_object_counts)} snapshots "
        f"containing {len(all_obj_ids)} object classes"
    )


    # Sample the snapshots to meet the target number of instances per object class
    selected_by_object, global_counts, selected_keys = select_snapshots_with_global_quota(
        object_to_snapshot_keys=object_to_snapshot_keys,
        snapshot_object_counts=snapshot_object_counts,
        target_samples=args.target_samples,
        rng=rng,
    )

    # Save the new dataset
    out_path = f"{args.in_path}_{args.out_prefix}"
    os.makedirs(out_path, exist_ok=True,)

    trigger_obj_ids = sorted(selected_by_object.keys())
    for obj_id in tqdm(trigger_obj_ids, desc="Processing selected snapshots", total=len(trigger_obj_ids)):
        selected_snapshot_keys = selected_by_object[obj_id]

        obj_scene_name = os.path.join(out_path, f"{obj_id:06d}")
        images_dir = os.path.join(obj_scene_name, "images")
        masks_dir = os.path.join(obj_scene_name, "images_masks")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        obj_snapshots = {}
        scene_objects = set()

        for idx, snapshot_key in enumerate(selected_snapshot_keys):
            scene, _ = snapshot_key

            snapshot_dict = all_snapshots[snapshot_key]

            rgb_img_path = os.path.join(args.in_path, scene, "images", snapshot_dict["rgb_img_filename"])
            depth_img_path = os.path.join(args.in_path, scene, "images", snapshot_dict["depth_img_filename"])
            mask_img_path = os.path.join(args.in_path, scene, "images_masks", snapshot_dict["mask_img_filename"])
            instances_mask_img_path = os.path.join(args.in_path, scene, "images_masks", snapshot_dict["instances_mask_img_filename"])

            rgb_filename = f"{idx:06d}_rgb.png"
            depth_filename = f"{idx:06d}_depth.png"
            mask_filename = f"{idx:06d}_mask.png"
            instances_mask_filename = f"{idx:06d}_instances_mask.png"

            shutil.copy(rgb_img_path, os.path.join(images_dir, rgb_filename))
            shutil.copy(depth_img_path, os.path.join(images_dir, depth_filename))
            shutil.copy(mask_img_path, os.path.join(masks_dir, mask_filename))
            shutil.copy(instances_mask_img_path, os.path.join(masks_dir, instances_mask_filename))

            new_snapshot_dict = snapshot_dict.copy()

            new_snapshot_dict["rgb_img_filename"] = rgb_filename
            new_snapshot_dict["depth_img_filename"] = depth_filename
            new_snapshot_dict["mask_img_filename"] = mask_filename
            new_snapshot_dict["instances_mask_img_filename"] = instances_mask_filename

            obj_snapshots[idx] = new_snapshot_dict

            scene_objects.update(snapshot_dict["objects_names"])

        io_utils.save_dict(obj_snapshots, "snapshots.yml", obj_scene_name)

        scene_info = {
            "objects_name": sorted(scene_objects),
            "snapshots_count": len(obj_snapshots),
        }
        io_utils.save_dict(scene_info, "scene_info.yml", obj_scene_name)


    print()
    print("Selection summary")
    print("-----------------")

    print(f"Selected source snapshots: {len(selected_keys)}/{len(snapshot_object_counts)}")
    reached_target = sum(
        1
        for obj_id in all_obj_ids
        if global_counts[obj_id] >= args.target_samples
    )
    print(f"Reached target for {reached_target}/{len(all_obj_ids)} objects")

    print()
    print("Final per-object counts:")
    for obj_id in all_obj_ids:
        count = global_counts[obj_id]
        delta = count - args.target_samples

        if delta == 0:
            status = "exact"
        elif delta > 0:
            status = f"+{delta}"
        else:
            status = str(delta)

        print(
            f"  obj {obj_id:2d}: "
            f"{count:6d} "
            f"(target={args.target_samples}, "
            f"delta={status})"
        )

    underflow_objects = sorted(
        [
            obj_id
            for obj_id in all_obj_ids
            if global_counts[obj_id] < args.target_samples
        ],
        key=lambda oid: (
            args.target_samples - global_counts[oid]
        ),
        reverse=True,
    )

    overflow_objects = sorted(
        [
            obj_id
            for obj_id in all_obj_ids
            if global_counts[obj_id] > args.target_samples
        ],
        key=lambda oid: (
            global_counts[oid] - args.target_samples
        ),
        reverse=True,
    )

    if underflow_objects:
        print()
        print("Objects below target:")

        for obj_id in underflow_objects:
            missing = (
                args.target_samples
                - global_counts[obj_id]
            )

            print(
                f"  obj {obj_id}: "
                f"-{missing} "
                f"(global={global_counts[obj_id]}, "
                f"target={args.target_samples})"
            )

    if overflow_objects:
        print()
        print("Objects above target due to snapshot co-occurrence:")

        for obj_id in overflow_objects:
            extra = (
                global_counts[obj_id]
                - args.target_samples
            )

            print(
                f"  obj {obj_id}: "
                f"+{extra} "
                f"(global={global_counts[obj_id]}, "
                f"target={args.target_samples})"
            )


if __name__ == "__main__":
    main()