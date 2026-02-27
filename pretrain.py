"""
pretrain.py

Original BENDR pretraining with our task-aware cluster loss added on top.
Without a label file this is identical to vanilla BENDR.

Pipeline: downloader.py to task_switch_identifi.py to gen_labels.py to HERE

New things added vs original:
  - ClusterLabelDataset: wraps DN3 datasets to append cluster IDs to each batch item
  - NEWBendingCollegeWav2Vec: drop-in replacement that adds cluster contrastive loss
  - --label-file / --num-clusters: only new CLI flags
  - --no-to1020: off by default, flip this when the GNN encoder is wired in
"""

import torch
import tqdm
import argparse
import numpy as np
import pandas as pd
import os
import glob
import json

from dn3_ext import NEWBendingCollegeWav2Vec, ConvEncoderBENDR, BENDRContextualizer
from dn3.configuratron import ExperimentConfig
from dn3.transforms.instance import To1020
from dn3.transforms.batch import RandomTemporalCrop
from gen_labels import load_label_file, epoch_label
from GNNencoder import GCNEncoder
from CNNencoder import CNNEncoder
from torch_geometric.utils import dense_to_sparse
from sklearn.neighbors import kneighbors_graph
from pathlib import Path
from torch.utils.data import ConcatDataset, Dataset

import mne
mne.set_log_level(False)


def safe_sfreq_for_dataset(toplevel, target_sfreq=256, epoch_secs=10):
    """
    Scans a BIDS dataset toplevel for the first loadable EEG file, reads its
    native sfreq from the header (no data loaded), then returns a safe target
    sfreq and corresponding sample count.

    Rules:
      - If native_sfreq == target_sfreq: use as-is
      - If native_sfreq is an integer multiple of target_sfreq AND the ratio
        is <= 4: safe to downsample, use target_sfreq
      - If the ratio > 4 or not integer: step up target_sfreq to the largest
        power-of-2 divisor of native_sfreq that is >= target_sfreq
      - If file is pre-epoched (multiple trials): returns None to signal skip

    Returns (safe_sfreq, n_samples) or (None, None) if no readable file found.
    """
    EEG_EXTENSIONS = ('.edf', '.bdf', '.fif', '.gdf', '.cnt', '.set')

    native_sfreq = None
    for root, _, files in os.walk(toplevel):
        # skip derivatives — they're often epoched
        if 'derivatives' in root:
            continue
        for f in files:
            if not f.endswith(EEG_EXTENSIONS):
                continue
            fpath = os.path.join(root, f)
            try:
                info = mne.io.read_raw(fpath, preload=False, verbose=False).info
                native_sfreq = info['sfreq']
                break
            except Exception:
                continue
        if native_sfreq is not None:
            break

    if native_sfreq is None:
        print(f"  [safe_sfreq] No readable EEG file found in {toplevel}")
        return None, None, None

    native_sfreq = int(native_sfreq)

    # also grab the lowpass reported in the header — used to detect when we need to pre-filter
    native_lowpass = info.get('lowpass', None)

    if native_sfreq == target_sfreq:
        chosen = target_sfreq
    elif native_sfreq % target_sfreq == 0 and (native_sfreq // target_sfreq) <= 4:
        # clean integer downsample, ratio small enough to be safe
        chosen = target_sfreq
    else:
        # find the largest power-of-2 divisor of native_sfreq >= target_sfreq
        # e.g. native=1024, target=256 -> ratio=4 -> ok
        #      native=1024, target=256 but ratio=4 already handled above
        #      native=1024, target=100 -> not integer -> step to 512
        chosen = native_sfreq
        while chosen > target_sfreq:
            half = chosen // 2
            if half >= target_sfreq and native_sfreq % half == 0:
                chosen = half
            else:
                break

    n_samples = int(chosen * epoch_secs)

    # check if the EDF header lowpass would still trigger DN3's aliasing guard
    # after choosing our sfreq: DN3 blocks if chosen < 2 * header_lowpass
    # fix: return a suggested lpf that will pass the check (chosen / 2 with some margin)
    suggested_lpf = None
    if native_lowpass is not None and chosen < 2 * native_lowpass:
        suggested_lpf = chosen / 2.0 * 0.9  # stay a bit under Nyquist
        print(f"  [safe_sfreq] {os.path.basename(toplevel)}: EDF header lowpass={native_lowpass}Hz "
              f"would block resampling to {chosen}Hz -> auto-setting lpf={suggested_lpf:.1f}Hz")
    elif chosen != target_sfreq:
        print(f"  [safe_sfreq] {os.path.basename(toplevel)}: native={native_sfreq}Hz, "
              f"requested={target_sfreq}Hz would alias -> using {chosen}Hz "
              f"({n_samples} samples per {epoch_secs}s epoch)")
    else:
        print(f"  [safe_sfreq] {os.path.basename(toplevel)}: native={native_sfreq}Hz -> {chosen}Hz OK")

    return chosen, n_samples, suggested_lpf


class ClusterLabelDataset(Dataset):
    """
    Wraps a DN3 dataset and appends a cluster label int to each sample.
    Turns (eeg,) into (eeg, cluster_id) where cluster_id is int64.
    -1 means the epoch is in a transition window and gets skipped by the cluster loss.
    Recordings not in the label dict still train normally on InfoNCE (all -1).
    """

    def __init__(self, base_dataset, label_dict, epoch_len):
        self.base      = base_dataset
        self.epoch_len = epoch_len
        self._labels   = self._build_labels(base_dataset, label_dict, epoch_len)

    @staticmethod
    def _build_labels(dataset, label_dict, epoch_len):
        #walk the DN3 thinker/session hierarchy in the same order the DataLoader will use
        #so that label indices stay aligned with batch indices throughout training
        n_total = len(dataset)
        labels  = np.full(n_total, -1, dtype=np.int64)
        idx     = 0

        def _find_key(abs_path):
            #task_switch_identifi.py stores relative paths, match by suffix
            normalised = str(abs_path).replace('\\', '/')
            for k in label_dict:
                if normalised.endswith(k.replace('\\', '/')):
                    return k
            return None

        try:
            for thinker in dataset.thinkers.values():
                for session in thinker.sessions.values():
                    filename = getattr(session, 'filename', None)
                    #_decimated_sequence_starts holds the start sample of each epoch window
                    starts   = getattr(session, '_decimated_sequence_starts', None)

                    if filename is None or starts is None:
                        idx += len(session)
                        continue

                    key   = _find_key(filename)
                    entry = label_dict.get(key) if key else None

                    for start in starts:
                        if idx >= n_total:
                            break
                        if entry is not None:
                            labels[idx] = epoch_label(
                                entry['labels'], entry['mask'], start, epoch_len)
                        idx += 1

        except Exception as e:
            #don't crash training if alignment fails, just fall back to InfoNCE-only
            tqdm.tqdm.write(f"Warning: cluster label alignment failed ({e}). Cluster loss inactive.")

        n_labelled = int((labels >= 0).sum())
        tqdm.tqdm.write(f"  {n_labelled}/{n_total} epochs assigned a cluster label.")
        return labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        #DN3 can return a bare tensor or a tuple depending on config, always grab EEG first
        eeg = sample[0] if isinstance(sample, (tuple, list)) else sample
        return eeg, torch.tensor(self._labels[idx], dtype=torch.long)

def adjacency_bids(dataset, subject, toplevel=None):
    """
    Build electrode adjacency graph from BIDS sidecar files.
    toplevel: path to the dataset root (e.g. ./on/ds003775).
              Falls back to ./data/on/<dataset> if not given.
    """
    if toplevel is not None:
        dataset_dir = toplevel
    else:
        base_dir = Path(__file__).parent
        dataset_dir = os.path.join(base_dir, "data", "on", dataset)

    subject_dir = os.path.join(dataset_dir, subject)

    search_json = os.path.join(subject_dir, "**", "*.json")
    search_tsv  = os.path.join(subject_dir, "**", "eeg", "*_channels.tsv")

    tsv_files  = glob.glob(search_tsv, recursive=True)
    json_files = glob.glob(search_json, recursive=True)

    if not json_files:
        raise FileNotFoundError(
            f"adjacency_bids: no JSON sidecar found under {subject_dir}. "
            f"Check that the dataset is downloaded and the toplevel path is correct. "
            f"Dataset dir searched: {dataset_dir}")

    with open(json_files[0], "r") as f:
        meta = json.load(f)
   
    def montageName(sch):
        scheme_clean = sch.lower().replace("-", "")
        montage_name = f"standard_{scheme_clean}"
        return montage_name
    
    scheme = (meta.get("EEGPlacementScheme") or "").lower()
    montage_name = montageName(scheme)

    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]

    df = pd.read_csv(tsv_files[0], sep="\t")
    ch_names = df["name"].dropna().astype(str).tolist()
    
    coords_2d = np.array([[pos[ch][0], pos[ch][1]] for ch in ch_names if ch in pos], dtype=float)
    kept_names = [ch for ch in ch_names if ch in pos]
    
    A = kneighbors_graph(coords_2d, n_neighbors=min(8, len(kept_names)-1),
                         mode="connectivity", include_self=True).toarray()
    A_matrix = torch.tensor(A, dtype=torch.float32)  
    edge_index, edge_weight = dense_to_sparse(A_matrix)

    return edge_index, edge_weight

def load_datasets(experiment, label_dict=None, epoch_len=2560, use_to1020=True, use_GNN=False):
    training       = []
    validation     = None
    total_thinkers = 0
    edge_index     = None
    edge_weight    = None

    target_sfreq = getattr(experiment, 'global_sfreq', 256)
    epoch_secs   = getattr(experiment, 'global_samples', epoch_len) / target_sfreq

    for name, ds in experiment.datasets.items():
        print("Constructing " + name)

        # auto-detect native sfreq and pick a safe downsample target
        toplevel = getattr(ds, 'toplevel', None)
        if toplevel is not None:
            safe_hz, safe_samples, suggested_lpf = safe_sfreq_for_dataset(
                toplevel, target_sfreq=target_sfreq, epoch_secs=epoch_secs)
            if safe_hz is None:
                # couldn't read a sample file (e.g. git-annex not unlocked, path missing)
                # don't skip — let DN3 attempt it and give its own error rather than silently dropping
                print(f"  [safe_sfreq] Could not pre-scan {name}, proceeding anyway.")
            else:
                # patch sfreq/samples if we had to step up to avoid aliasing
                if safe_hz != target_sfreq:
                    ds.sfreq    = safe_hz
                    ds.stride   = safe_samples
                    ds.samples  = safe_samples
                # patch lpf when the EDF header lowpass would trigger DN3's aliasing guard
                if suggested_lpf is not None and not hasattr(ds, 'lpf'):
                    ds.lpf = suggested_lpf

        dataset = ds.auto_construct_dataset()

        if use_to1020:
            dataset.add_transform(To1020())
        elif use_GNN:
            subj = "sub-001"
            edge_index, edge_weight = adjacency_bids(name, subj, toplevel=toplevel)

        is_validation = (hasattr(experiment, 'validation_dataset') and
                         experiment.validation_dataset == name)
        if is_validation:
            validation = dataset
            continue

        total_thinkers += len(dataset.get_thinkers())

        if label_dict is not None:
            dataset = ClusterLabelDataset(dataset, label_dict, epoch_len)

        training.append(dataset)

    print("Training BENDR using {} people's data across {} datasets.".format(
        total_thinkers, len(training)))

    if len(training) == 0:
        paths = {n: getattr(ds, 'toplevel', 'unknown') for n, ds in experiment.datasets.items()}
        raise RuntimeError(
            "No training datasets found. Things to check:\n"
            f"  1. Dataset paths exist on this machine: {paths}\n"
            "  2. 'validation_dataset' isn't set to the only dataset in the config.\n"
            "  3. If using git-annex (datalad BIDS), run: git -C <dataset_path> annex unlock .\n"
            "  4. If data is missing entirely, run: git -C <dataset_path> annex get ."
        )

    # edge_index/edge_weight are None when not using GNN (e.g. test runs with To1020)
    _ei = edge_index if use_GNN else None
    _ew = edge_weight if use_GNN else None
    return ConcatDataset(training), validation, total_thinkers, _ei, _ew


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrains a BENDR model.")
    parser.add_argument('--config', default="configs/pretraining.yml")
    parser.add_argument('--hidden-size', default=512, type=int)
    parser.add_argument('--resume', default=None, type=int,
                        help="Resume from this epoch checkpoint.")
    parser.add_argument('--num-workers', default=6, type=int)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-save-epochs', action='store_true')
    #our additions
    parser.add_argument('--label-file', default=None,
                        help="cluster_labels.npy from gen_labels.py. Activates cluster loss.")
    parser.add_argument('--num-clusters', default=3, type=int,
                        help="Should match what gen_labels.py was run with.")
    parser.add_argument('--no-to1020', action='store_true',
                        help="Skip To1020 remapping. Required when using --use-gnn.")
    parser.add_argument('--use-gnn', action='store_true',
                        help="Use CNN+GNN encoder instead of ConvEncoderBENDR.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    experiment = ExperimentConfig(args.config)

    #load cluster labels if provided, otherwise runs vanilla BENDR with InfoNCE only
    label_dict = None
    if args.label_file is not None:
        print(f"Loading cluster labels from {args.label_file} ...")
        label_dict = load_label_file(args.label_file)

    epoch_len  = getattr(experiment, 'global_samples', 256 * 10)
    use_GNN    = args.use_gnn
    use_to1020 = not args.no_to1020 and not use_GNN  # GNN handles its own channels

    training, validation, target_thinkers, edge_index, edge_weight = load_datasets(
        experiment, label_dict=label_dict, epoch_len=epoch_len, use_to1020=use_to1020, use_GNN=use_GNN)

    if use_GNN:
        cnn = CNNEncoder(
            output_channels=max(1, args.hidden_size // 4),
            kernel_sizes=(128, 64, 32),
            pool_sizes=(5, 3, 2),
            dropout=0.5,
            stride=1,
            padding="same",
        )
        gnn = GCNEncoder(
            nfeat=cnn.F,
            nhid=args.hidden_size,
            nout=args.hidden_size,
            dropout=getattr(experiment, "dropout", 0.5),
            pool_ratio=0.9,
        )

        class Encoder(torch.nn.Module):
            def __init__(self, cnn, gnn, encoder_h):
                super().__init__()
                self.cnn = cnn
                self.gnn = gnn
                self.encoder_h = encoder_h
                self.edge_index  = None
                self.edge_weight = None

            def forward(self, x):
                h = self.cnn(x)
                _, z_seq = self.gnn(h, self.edge_index, self.edge_weight)
                return z_seq

            def save(self, path):
                torch.save(self.state_dict(), path)

            def load(self, path):
                self.load_state_dict(torch.load(path))

            def description(self, sfreq, samples):
                return f"CNN+GNN Encoder | sfreq={sfreq} | samples={samples} | hidden={self.encoder_h}"

        encoder = Encoder(cnn, gnn, encoder_h=args.hidden_size)
        encoder.edge_index  = edge_index
        encoder.edge_weight = edge_weight
    else:
        # default: ConvEncoderBENDR with To1020 fixed 21 channels
        n_channels = len(To1020.EEG_20_div) + 1
        encoder = ConvEncoderBENDR(n_channels, encoder_h=args.hidden_size)
    tqdm.tqdm.write(encoder.description(
        getattr(experiment, 'global_sfreq', 256),
        getattr(experiment, 'global_samples', 2560)))

    contextualizer = BENDRContextualizer(
        encoder.encoder_h,
        layer_drop=experiment.bending_college_args.layer_drop)

    if args.resume is not None:
        encoder.load('checkpoints/encoder_epoch_{}.pt'.format(args.resume))
        contextualizer.load('checkpoints/contextualizer_epoch_{}.pt'.format(args.resume))

    #use_cluster_loss only does something when label_dict is not None
    process = NEWBendingCollegeWav2Vec(
        encoder, contextualizer,
        use_cluster_loss = label_dict is not None,
        num_clusters     = args.num_clusters,
        **experiment.bending_college_args)

    tqdm.tqdm.write(process.description(getattr(experiment, 'global_samples', 2560)))

    # DN3 config objects carry internal _-prefixed keys, strip them before unpacking
    def _cfg(obj):
        return {k: v for k, v in vars(obj).items() if not k.startswith('_')}

    process.set_optimizer(
        torch.optim.Adam(process.parameters(), **_cfg(experiment.optimizer_params)))
    process.add_batch_transform(
        RandomTemporalCrop(max_crop_frac=experiment.augmentation_params.batch_crop_frac))

    def epoch_checkpoint(metrics):
        if not args.no_save and not args.no_save_epochs:
            tqdm.tqdm.write("Saving epoch checkpoint...")
            encoder.save('checkpoints/encoder_epoch_{}.pt'.format(metrics['epoch']))
            contextualizer.save('checkpoints/contextualizer_epoch_{}.pt'.format(metrics['epoch']))

    def simple_checkpoint(metrics):
        #widen mask span when the model is performing well enough, keeps the task hard
        if metrics is not None \
                and metrics['Accuracy'] > experiment.mask_threshold \
                and metrics['Mask_pct'] < experiment.mask_pct_max:
            process.mask_span = int(process.mask_span * experiment.mask_inflation)
            tqdm.tqdm.write("Increased mask span to {} samples".format(process.mask_span))
        if not args.no_save:
            tqdm.tqdm.write("Saving...")
            encoder.save('checkpoints/encoder.pt')
            contextualizer.save('checkpoints/contextualizer.pt')

    #save initial weights before any training so we always have a fallback checkpoint
    simple_checkpoint(None)

    process.fit(training,
                epoch_callback     = epoch_checkpoint,
                num_workers        = args.num_workers,
                validation_dataset = validation,
                resume_epoch       = args.resume,
                log_callback       = simple_checkpoint,
                **_cfg(experiment.training_params))

    if validation is not None:
        print(process.evaluate(validation))

    if not args.no_save:
        tqdm.tqdm.write("Saving best model...")
        encoder.save('checkpoints/encoder_best_val.pt')
        contextualizer.save('checkpoints/contextualizer_best_val.pt')
