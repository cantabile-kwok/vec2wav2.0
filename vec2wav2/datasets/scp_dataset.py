# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# Modified by Yiwei Guo, 2024

"""Dataset modules based on kaldi-style scp files."""

import logging
import random
import copy
from multiprocessing import Manager

import kaldiio
import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm
from vec2wav2.utils import HDF5ScpLoader
from vec2wav2.utils import NpyScpLoader


def _get_feats_scp_loader(feats_scp):
    # read the first line of feats.scp file
    with open(feats_scp) as f:
        key, value = f.readlines()[0].replace("\n", "").split()

    # check scp type
    if ":" in value:
        value_1, value_2 = value.split(":")
        if value_1.endswith(".ark"):
            # kaldi-ark case: utt_id_1 /path/to/utt_id_1.ark:index
            return kaldiio.load_scp(feats_scp)
        elif value_1.endswith(".h5"):
            # hdf5 case with path in hdf5: utt_id_1 /path/to/utt_id_1.h5:feats
            return HDF5ScpLoader(feats_scp)
        else:
            raise ValueError("Not supported feats.scp type.")
    else:
        if value.endswith(".h5"):
            # hdf5 case without path in hdf5: utt_id_1 /path/to/utt_id_1.h5
            return HDF5ScpLoader(feats_scp)
        elif value.endswith(".npy"):
            # npy case: utt_id_1 /path/to/utt_id_1.npy
            return NpyScpLoader(feats_scp)
        else:
            raise ValueError("Not supported feats.scp type.")


class AudioMelSCPDataset(Dataset):
    """PyTorch compatible audio and feat dataset based on kaldi-stype scp files."""

    def __init__(
        self,
        wav_scp,
        vqidx_scp,
        mel_scp,
        prompt_scp,
        utt2num_frames=None,
        segments=None,
        batch_frames=None,
        batch_size=None,
        min_num_frames=None,
        max_num_frames=None,
        return_utt_id=False,
        return_sampling_rate=False,
        allow_cache=False,
        length_tolerance=2,
        prompt_fold_by_2=True
    ):
        """Initialize dataset.

        Args:
            wav_scp (str): Kaldi-style wav.scp file.
            vqidx_scp (str): Kaldi-style fests.scp file.
            mel_scp (str): Kaldi-style fests.scp file.
            segments (str): Kaldi-style segments file.
            min_num_frames (int): Threshold to remove short feature files.
            max_num_frames (int): Threshold to remove long feature files.
            return_utt_id (bool): Whether to return utterance id.
            return_sampling_rate (bool): Whether to return sampling rate.
            allow_cache (bool): Whether to allow cache of the loaded files.
            prompt_fold_by_2 (bool): if true, then prompt have half the length of vqidx sequence.

        """
        # load scp as lazy dict
        self.audio_loader = kaldiio.load_scp(wav_scp, segments=segments)
        self.vqidx_loader = _get_feats_scp_loader(vqidx_scp)
        self.mel_loader = _get_feats_scp_loader(mel_scp)

        self.prompt_loader = _get_feats_scp_loader(prompt_scp)

        self.utt_ids = list(self.mel_loader.keys())
        self.return_utt_id = return_utt_id
        self.return_sampling_rate = return_sampling_rate
        self.allow_cache = allow_cache

        utt2num_frames_loader = None
        if utt2num_frames is not None:
            with open(utt2num_frames, 'r') as f:
                utt2num_frames_loader = dict([(x.split()[0], int(x.split()[1])) for x in f.readlines()])
        else:
            utt2num_frames_loader = dict([(k, mel.shape[0]) for k, mel in self.mel_loader.items()])

        self.utt2num_frames_loader = utt2num_frames_loader

        # filter by threshold
        if (min_num_frames or max_num_frames) is not None:
            mel_lengths = [utt2num_frames_loader[key] for key in self.utt_ids]
            idxs = [
                idx
                for idx in range(len(self.utt_ids))
                if (min_num_frames and mel_lengths[idx] >= min_num_frames) and (max_num_frames and mel_lengths[idx] <= max_num_frames)
            ]
            if len(self.utt_ids) != len(idxs):
                logging.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(self.utt_ids)} -> {len(idxs)})."
                )
            self.utt_ids = [self.utt_ids[idx] for idx in idxs]

        # batchify
        if batch_frames is not None:
            self.batches = self.batchify(utt2num_frames_loader, batch_frames=batch_frames)
        elif batch_size is not None:
            self.batches = self.batchify(utt2num_frames_loader, batch_size=batch_size)
        else:
            self.batches = [[utt_id] for utt_id in self.utt_ids]

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.dict()
        self.length_tolerance = length_tolerance
        if prompt_fold_by_2:
            self.prompt_len_factor = 2
        else:
            self.prompt_len_factor = 1

    def batchify(self, utt2num_frames_loader, batch_frames=None, batch_size=None, min_batch_size=1, drop_last=True):

        assert batch_size is None or batch_size > min_batch_size

        batches = []
        batch = []
        accum_num_frames = 0
        utt_ids_set = set(self.utt_ids)
        for utt_id, mel_length in tqdm(sorted(list(utt2num_frames_loader.items()), key=lambda x: x[1], reverse=True)):
            if utt_id not in utt_ids_set:
                continue
            if (batch_frames is not None and accum_num_frames + mel_length > batch_frames and len(batch) > min_batch_size) or (batch_size is not None and len(batch) == batch_size):
                batches.append(batch)
                batch = []
                accum_num_frames = 0
            batch.append(utt_id)
            accum_num_frames += mel_length
        if len(batch) > min_batch_size and not drop_last:
            batches.append(batch)
        return batches

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray or tuple: Audio signal (T,) or (w/ sampling rate if return_sampling_rate = True).
            ndarrays: Features (T', C).

        """
        batch = self.batches[idx]
        batch_items = []

        for utt_id in batch:
            if self.allow_cache and self.caches.get(utt_id) is not None:
                items = self.caches[utt_id]
            else:
                fs, audio = self.audio_loader[utt_id]
                mel = self.mel_loader[utt_id]
                prompt = self.prompt_loader[utt_id]
                vqidx = self.vqidx_loader[utt_id]

                min_len = min(len(mel), len(vqidx), len(prompt)*self.prompt_len_factor)
                assert ((abs(len(mel) - min_len) <= self.length_tolerance) and
                        (abs(len(vqidx) - min_len) <= self.length_tolerance) and
                        (abs(len(prompt)*self.prompt_len_factor - min_len) <= self.length_tolerance)), \
                    f"Audio feature lengths difference exceeds length tolerance for {utt_id}"
                mel, vqidx, prompt = mel[:min_len], vqidx[:min_len], prompt[:min_len//self.prompt_len_factor]

                # normalize audio signal to be [-1, 1]
                audio = audio.astype(np.float32)
                audio /= 1 << (16 - 1)  # assume that wav is PCM 16 bit

                if self.return_sampling_rate:
                    audio = (audio, fs)

                if self.return_utt_id:
                    items = utt_id, audio, vqidx, mel, prompt
                else:
                    items = audio, vqidx, mel, prompt

                if self.allow_cache:
                    self.caches[utt_id] = items

            batch_items.append(items)

        return batch_items

    def __len__(self):
        """Return dataset length.
        Returns:
            int: The length of dataset.
        """
        return len(self.batches)


class MelSCPDataset(Dataset):
    """PyTorch compatible feat dataset based on kaldi-stype scp files."""

    def __init__(
        self,
        vqidx_scp,
        prompt_scp,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            vqidx_scp (str): Kaldi-style fests.scp file.
            prompt_scp (str): Kaldi-style scp file. In this file, every utt is associated with its prompt's mel-spectrogram.
            min_num_frames (int): Threshold to remove short feature files.
            max_num_frames (int): Threshold to remove long feature files.
            return_utt_id (bool): Whether to return utterance id.
            allow_cache (bool): Whether to allow cache of the loaded files.
        """
        # load scp as lazy dict
        vqidx_loader = _get_feats_scp_loader(vqidx_scp)
        self.prompt_loader = _get_feats_scp_loader(prompt_scp)
        # self.prompt_loader = dict()
        # with open(prompt_scp, 'r') as fr:
            # for line in fr.readlines():
                # terms = line.strip().split()
                # self.prompt_loader[terms[0]] = terms[1]
        vqidx_keys = list(set(self.prompt_loader.keys()) & set(vqidx_loader.keys()))

        # NOTE: this dataset does not apply filtering, because it is usually used for decoding
        
        self.vqidx_loader = vqidx_loader
        self.utt_ids = vqidx_keys
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.utt_ids))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        vqidx = self.vqidx_loader[utt_id].astype(int)

        # prompt = torch.load(self.prompt_loader[utt_id]).float().numpy()
        prompt = self.prompt_loader[utt_id]

        if self.return_utt_id:
            items = utt_id, vqidx, prompt
        else:
            items = vqidx, prompt

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.utt_ids)
