"""
gen_labels.py

Takes the task switch JSON from task_switch_identifi.py and converts it into
per-sample cluster label arrays that pretrain.py can load and feed into the loss.

Pipeline: task_switch_identifi.py -> task_switch_labels.json -> HERE -> cluster_labels.npy -> pretrain.py

Input JSON format:
    [{"name": "relative/file.edf", "task_switch": [[start, end], ...]}, ...]

Output .npy format (dict keyed by filename):
    {"labels": int64 array (n_samples,), "mask": float32 array (n_samples,)}
    label = cluster ID per sample, -1 during transition windows
    mask  = 1.0 usable, 0.0 transition

Leopold Ehrlich
February 2026
"""

import json
import argparse
import numpy as np
from pathlib import Path


#these should match what the DN3 config uses (samples and sfreq fields)
SAMPLE_RATE = 256
EPOCH_LEN   = SAMPLE_RATE * 10  # 10 second epochs = 2560 samples


def create_labels(seg_tuples, total_samples):
    """
    Build per-sample label and mask arrays for one recording.

    seg_tuples are the TRANSITION windows from task_switch_identifi.py,
    not the stable segments. Stable segments between transitions get
    sequential cluster IDs (0, 1, 2...). Transition windows stay at -1.
    """
    #start everything as transition/unknown, fill in stable segments below
    label = np.full(total_samples, -1, dtype=np.int64)
    mask  = np.zeros(total_samples, dtype=np.float32)

    prev_end = 0
    for task_id, (start, end) in enumerate(seg_tuples):
        #label the stable region that sits before this transition
        if start > prev_end:
            label[prev_end:start] = task_id
            mask[prev_end:start]  = 1.0
        #transition window stays at label=-1, mask=0
        prev_end = end

    #label the final stable region after the last transition
    if prev_end < total_samples:
        label[prev_end:] = len(seg_tuples)
        mask[prev_end:]  = 1.0

    return label, mask


def epoch_label(label, mask, epoch_start, epoch_len=EPOCH_LEN):
    """
    Collapse per-sample labels to a single integer for one epoch window.

    Returns -1 if more than 50% of the epoch is in a transition window.
    Otherwise returns the majority-vote cluster ID of the stable samples.
    This is what ClusterLabelDataset in pretrain.py calls for each epoch.
    """
    epoch_end  = min(epoch_start + epoch_len, len(label))
    seg_labels = label[epoch_start:epoch_end]
    seg_mask   = mask[epoch_start:epoch_end]

    #reject epochs that are mostly transition, too noisy to cluster reliably
    if seg_mask.mean() < 0.5:
        return -1

    #only vote over the stable non-transition samples
    valid = seg_labels[seg_mask > 0]
    valid = valid[valid >= 0]
    if len(valid) == 0:
        return -1

    #majority vote, whichever cluster ID appears most wins
    counts = np.bincount(valid.astype(np.intp))
    return int(np.argmax(counts))


def extract_switches(task_switch_path):
    """
    Load the task switch JSON written by task_switch_identifi.py.
    JSON stores tuples as arrays so we convert back with tuple(seg).
    Recordings with no detected switches have an empty list, that's fine.
    """
    with open(task_switch_path, 'r') as f:
        data = json.load(f)
    return {
        entry['name']: [tuple(seg) for seg in entry['task_switch']]
        for entry in data
    }


def gen_label_vec(task_switch_path, output_path, total_samples_map=None):
    """
    Build and save label/mask arrays for every recording in the task switch file.
    Run this once offline before training.

    total_samples_map is an optional {filename: n_samples} dict.
    If not given, falls back to EPOCH_LEN which is only useful for quick tests.
    """
    switches = extract_switches(task_switch_path)
    results  = {}

    for file_name, segs in switches.items():
        n = (total_samples_map or {}).get(file_name, EPOCH_LEN)
        labels, mask = create_labels(segs, n)
        results[file_name] = {'labels': labels, 'mask': mask}

        n_labelled = int((labels >= 0).sum())
        print(f"  {file_name}: {len(segs)} transitions, "
              f"{n_labelled}/{n} labelled ({100*n_labelled/max(n,1):.1f}%)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, results)
    print(f"Saved cluster labels for {len(results)} files -> {output_path}")
    return results


def load_label_file(label_path):
    """
    Load a pre-computed label file from disk.
    allow_pickle=True is needed because the .npy stores a dict of dicts.
    .item() unwraps the 0-d numpy array wrapper np.save puts around Python objects.
    """
    return np.load(label_path, allow_pickle=True).item()


def get_epoch_cluster_label(label_dict, file_name, epoch_start_sample,
                             epoch_len=EPOCH_LEN):
    """
    Convenience function to look up one epoch's cluster label.
    Returns -1 if the file wasn't in the label dict.
    """
    entry = label_dict.get(file_name)
    if entry is None:
        return -1
    return epoch_label(entry['labels'], entry['mask'], epoch_start_sample, epoch_len)


def build_epoch_label_array(label_dict, file_name, n_samples,
                             epoch_len=EPOCH_LEN, step=None):
    """
    Pre-compute cluster labels for every epoch in a recording as a flat array.
    More efficient than calling get_epoch_cluster_label in a loop.
    step defaults to epoch_len (no overlap), matching the DN3 config stride.
    """
    if step is None:
        step = epoch_len

    entry = label_dict.get(file_name)
    if entry is None:
        #no label data for this file, fill with -1 so it still trains on InfoNCE
        n_epochs = max(1, (n_samples - epoch_len) // step + 1)
        return np.full(n_epochs, -1, dtype=np.int64)

    starts    = np.arange(0, n_samples - epoch_len + 1, step)
    ep_labels = np.array([
        epoch_label(entry['labels'], entry['mask'], s, epoch_len)
        for s in starts
    ], dtype=np.int64)
    return ep_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert task_switch_labels.json to cluster_labels.npy")
    parser.add_argument('--task-switch', default='task_switch_labels.json',
                        help="JSON from task_switch_identifi.py")
    parser.add_argument('--output', default='processed_data/cluster_labels.npy',
                        help="Output .npy file path")
    args = parser.parse_args()

    print(f"Loading task switches from {args.task_switch} ...")
    gen_label_vec(args.task_switch, args.output)
