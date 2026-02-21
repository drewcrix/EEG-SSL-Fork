"""
pretrain.py

Original BENDR pretraining with our task-aware cluster loss added on top.
Without a label file this is identical to vanilla BENDR.

Pipeline: downloader.py -> task_switch_identifi.py -> gen_labels.py -> HERE

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

from dn3_ext import NEWBendingCollegeWav2Vec, ConvEncoderBENDR, BENDRContextualizer
from dn3.configuratron import ExperimentConfig
from dn3.transforms.instance import To1020
from dn3.transforms.batch import RandomTemporalCrop
from gen_labels import load_label_file, epoch_label

from torch.utils.data import ConcatDataset, Dataset

import mne
mne.set_log_level(False)


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


def load_datasets(experiment, label_dict=None, epoch_len=2560, use_to1020=True):
    #To1020 maps all electrode layouts to the standard 21-channel 10-20 montage
    #ConvEncoderBENDR needs exactly 21 channels, remove this once GNN encoder is in
    training       = []
    validation     = None
    total_thinkers = 0

    for name, ds in experiment.datasets.items():
        print("Constructing " + name)
        dataset = ds.auto_construct_dataset()

        if use_to1020:
            dataset.add_transform(To1020())

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
    return ConcatDataset(training), validation, total_thinkers


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
    #flip this when GNN encoder is ready and To1020 can come out
    parser.add_argument('--no-to1020', action='store_true',
                        help="Skip To1020 remapping, only needed for GNN encoder (future).")
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
    use_to1020 = not args.no_to1020

    training, validation, target_thinkers = load_datasets(
        experiment, label_dict=label_dict, epoch_len=epoch_len, use_to1020=use_to1020)

    #To1020 gives 21 channels (EEG_20_div + 1), update this when GNN encoder is in
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

    process.set_optimizer(
        torch.optim.Adam(process.parameters(), **experiment.optimizer_params))
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
                **experiment.training_params)

    print(process.evaluate(validation))

    if not args.no_save:
        tqdm.tqdm.write("Saving best model...")
        encoder.save('checkpoints/encoder_best_val.pt')
        contextualizer.save('checkpoints/contextualizer_best_val.pt')
