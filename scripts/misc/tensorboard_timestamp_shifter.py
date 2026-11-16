# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

"""
Script used to shift the "wall_time" timestamps of logged tensorboard tfevents file to help with data visualization and comparison.    
"""
import tensorflow as tf
import glob
import os


def shift_timestamps(input_file, output_file, shift_seconds):
    """
    Shift the timestamps of the TensorBoard records in a TFRecord file.

    Parameters:
        input_file (str): Path to the input TFRecord file.
        output_file (str): Path to the output TFRecord file.
        shift_seconds (int or float): Number of seconds to shift the timestamps.
    """
    dataset = tf.data.TFRecordDataset(input_file)

    with tf.io.TFRecordWriter(output_file) as writer:
        for raw_record in dataset:
            event = tf.compat.v1.Event()
            event.ParseFromString(raw_record.numpy())
            
            event.wall_time += shift_seconds
            
            writer.write(event.SerializeToString())


if __name__ == "__main__":
    # Base path to the directory containing the training runs outputs
    base_path = os.path.expandvars("${WS_DIR}/runs/obj2_transforms")
    
    # Number of seconds to shift the timestamps
    #   3600*(-2) --> shift 2 hours in the past 
    shift_seconds = 3600*(5) 
    
    # Interval of runs that we want to shift:
    #   start=0 and end=None --> all runs are shifted
    #   start=3 and end=9    --> shift from the fourth to the ninth run
    start, end = (0, None)


    # ------------------------------------------------ #


    # Load all tfevents
    tfevents = sorted(glob.glob(os.path.join(base_path, "*/*.tfevents*")))
    
    # Select the ones in the given range
    tfevents = tfevents[start:end]

    print("")
    print("")
    print("Selected tensorboard tfevents:")
    for event_name in tfevents: print("   ", event_name)
    if input(f"Do you want to shift by {shift_seconds/3600} hours the timestamps of the above {len(tfevents)} tfevents? y/n ").lower() == "y":    
        for event_name in tfevents:
            # Rename the original tensorboard log file
            event_renamed = event_name + "_old"
            os.rename(event_name, event_renamed)

            # Shift the timestamps
            shift_timestamps(event_renamed, event_name, shift_seconds)

            # Remove the old file
            os.remove(event_renamed)

    print("")
    print("")
    print("Timestamps shift completed!")