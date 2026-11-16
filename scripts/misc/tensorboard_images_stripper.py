# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import argparse
import tensorflow as tf

from glob import glob
from tqdm import tqdm

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.compat.proto import event_pb2


def strip_images_from_eventfile(input_path, keep_backup: bool = False):
    backup_path = input_path.replace("events.out.tfevents", "events.out.t_fevents")
    if os.path.exists(backup_path):
        tqdm.write("Backup file already exists, skipping...")
        return

    os.rename(input_path, backup_path)

    loader = EventFileLoader(backup_path)

    try:
        with tf.io.TFRecordWriter(input_path) as writer:
            for event in tqdm(
                loader.Load(),
                desc="Processing events",
                unit="event",
                leave=False,
            ):
                new_event = event_pb2.Event()
                new_event.CopyFrom(event)

                if new_event.HasField("summary"):
                    kept_values = []

                    for value in new_event.summary.value:
                        # Old-style image summary.
                        if value.HasField("image"):
                            continue

                        # New-style TensorBoard image summary.
                        if value.metadata.plugin_data.plugin_name == "images":
                            continue

                        kept_values.append(value)

                    del new_event.summary.value[:]
                    new_event.summary.value.extend(kept_values)

                writer.write(new_event.SerializeToString())

    except Exception:
        # Restore the original file if processing fails.
        if os.path.exists(input_path):
            os.remove(input_path)
        os.rename(backup_path, input_path)
        raise

    original_size = os.path.getsize(backup_path)
    stripped_size = os.path.getsize(input_path)

    tqdm.write(
        f"Done: "
        f"{original_size / 1024**2:.1f} MB -> "
        f"{stripped_size / 1024**2:.1f} MB"
    )

    # Remove original only after successful conversion.
    if not keep_backup:
        os.remove(backup_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", "-bd", default=None, required=True, help="Base directory containing the event files to process.")
    parser.add_argument("--keep_backup", action="store_true", help="Keep a backup of the original event files.") 
    args = parser.parse_args()

    path = os.path.expandvars(os.path.join(args.base_dir, "**/events.out.tfevents.*"))

    files = sorted(glob(path, recursive=True))
    for input_path in tqdm(files, desc="Event files", unit="file"):
        tqdm.write(f"Stripping images from {input_path}")
        strip_images_from_eventfile(input_path, keep_backup=args.keep_backup)