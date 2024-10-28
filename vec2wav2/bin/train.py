#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# Modified by Yiwei Guo, 2024

"""Train vec2wav."""

import argparse
import logging
import os
import sys
import random

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import vec2wav2
import vec2wav2.models
import vec2wav2.optimizers
from torch.utils.data.distributed import DistributedSampler

from vec2wav2.datasets import AudioMelSCPDataset
from vec2wav2.layers import PQMF
from vec2wav2.losses import DiscriminatorAdversarialLoss
from vec2wav2.losses import FeatureMatchLoss
from vec2wav2.losses import GeneratorAdversarialLoss
from vec2wav2.losses import MelSpectrogramLoss
from vec2wav2.losses import MultiResolutionSTFTLoss
from vec2wav2.utils import crop_seq, load_feat_codebook, idx2vec

from vec2wav2.utils.espnet_utils import pad_list, make_non_pad_mask

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


def set_loglevel(verbose):
    # set logger
    if verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")


class Trainer(object):
    """Customized trainer module for Parallel WaveGAN training."""

    def __init__(
            self,
            steps,
            epochs,
            data_loader,
            sampler,
            model,
            criterion,
            optimizer,
            scheduler,
            config,
            device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contain "train" and "dev" loaders.
            model (dict): Dict of models. It must contain "generator" and "discriminator" models.
            criterion (dict): Dict of criteria. It must contain "stft" and "mse" criteria.
            optimizer (dict): Dict of optimizers. It must contain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

        # load vq codebook
        feat_codebook_path = self.config["vq_codebook"]

        self.feat_codebook, self.feat_codebook_numgroups = load_feat_codebook(np.load(feat_codebook_path, allow_pickle=True), device)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(initial=self.steps, total=self.config["train_max_steps"], desc="[train]")
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
                "discriminator": self.model["discriminator"].module.state_dict(),
            }
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
                "discriminator": self.model["discriminator"].state_dict(),
            }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model["generator"].module.load_state_dict(
                state_dict["model"]["generator"]
            )
            self.model["discriminator"].module.load_state_dict(
                state_dict["model"]["discriminator"]
            )
        else:
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].load_state_dict(
                state_dict["model"]["discriminator"]
            )
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])
            self.optimizer["discriminator"].load_state_dict(state_dict["optimizer"]["discriminator"])
            self.scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])
            self.scheduler["discriminator"].load_state_dict(state_dict["scheduler"]["discriminator"])
    
    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        vqidx, _, mel, prompt, y, xlens, prompt_lens = batch
        vqidx = vqidx.to(self.device)
        mel = mel.to(self.device)
        prompt = prompt.to(self.device)
        vqvec = idx2vec(self.feat_codebook, vqidx, self.feat_codebook_numgroups)  # (B, L, D)
        y = y.unsqueeze(-2).to(self.device)  # (B, 1, T)

        # build mask
        mask = make_non_pad_mask(xlens).to(self.device)  # (B, L)
        prompt_mask = make_non_pad_mask(prompt_lens).to(self.device)  # (B, L_prompt)

        # crop wav sequence
        crop_xlen = min(self.config["crop_max_frames"], min(xlens))
        x_offsets = [np.random.randint(0, l - crop_xlen + 1) for l in xlens]
        crop_ylen = crop_xlen * self.config["hop_size"]
        y_offsets = [o * self.config["hop_size"] for o in x_offsets]
        y = crop_seq(y, y_offsets, crop_ylen)

        #######################
        #      Generator      #
        #######################
        if self.steps > self.config.get("generator_train_start_steps", 0):
            mel_, _, y_ = self.model["generator"](vqvec, prompt, mask, prompt_mask, crop_xlen, x_offsets)  # (B, L, 80), (B, C, T)

            # initialize
            gen_loss, aux_loss = 0.0, 0.0

            # frontend mel prediction loss
            if self.steps <= self.config.get("frontend_mel_prediction_stop_steps", 0):
                frontend_mel_pred_loss = F.l1_loss(torch.masked_select(mel, mask.unsqueeze(-1)),
                                                   torch.masked_select(mel_, mask.unsqueeze(-1)))
                self.total_train_loss["train/frontend_mel_pred_loss"] += frontend_mel_pred_loss.item()
                gen_loss += self.config["lambda_frontend_mel_prediction"] * frontend_mel_pred_loss

            # multi-resolution sfft loss
            if self.config["use_stft_loss"]:
                sc_loss, mag_loss = self.criterion["stft"](y_, y)
                aux_loss += sc_loss + mag_loss
                self.total_train_loss["train/spectral_convergence_loss"] += sc_loss.item()
                self.total_train_loss["train/log_stft_magnitude_loss"] += mag_loss.item()

            # subband multi-resolution stft loss
            if self.config["use_subband_stft_loss"]:
                aux_loss *= 0.5  # for balancing with subband stft loss
                y_mb = self.criterion["pqmf"].analysis(y)
                y_mb_ = self.criterion["pqmf"].analysis(y_)
                sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
                aux_loss += 0.5 * (sub_sc_loss + sub_mag_loss)
                self.total_train_loss["train/sub_spectral_convergence_loss"] += sub_sc_loss.item()
                self.total_train_loss["train/sub_log_stft_magnitude_loss"] += sub_mag_loss.item()

            # mel spectrogram loss
            if self.config["use_mel_loss"]:
                mel_loss = self.criterion["mel"](y_, y)
                aux_loss += mel_loss
                self.total_train_loss["train/mel_loss"] += mel_loss.item()

            # weighting aux loss
            gen_loss += self.config.get("lambda_aux", 1.0) * aux_loss

            # adversarial loss
            if self.steps > self.config["discriminator_train_start_steps"]:
                p_ = self.model["discriminator"](y_)
                adv_loss = self.criterion["gen_adv"](p_)
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()

                # feature matching loss
                if self.config["use_feat_match_loss"]:
                    # no need to track gradients
                    with torch.no_grad():
                        p = self.model["discriminator"](y)
                    fm_loss = self.criterion["feat_match"](p_, p)
                    self.total_train_loss["train/feature_matching_loss"] += fm_loss.item()
                    adv_loss += self.config["lambda_feat_match"] * fm_loss

                # add adversarial loss to generator loss
                gen_loss += self.config["lambda_adv"] * adv_loss

            self.total_train_loss["train/generator_loss"] += gen_loss.item()

            # update generator
            self.optimizer["generator"].zero_grad()
            gen_loss.backward()
            if self.config["generator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["generator"].parameters(),
                    self.config["generator_grad_norm"],
                )
            self.optimizer["generator"].step()
            self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                # logging.info(f"{vqvec.shape, prompt.shape, mask.shape, prompt_mask.shape}")
                _, _, y_ = self.model["generator"](vqvec, prompt, mask, prompt_mask, crop_xlen, x_offsets)  # (B, L, 80), (B, C, T)

            if self.config["generator_params"]["out_channels"] > 1:
                y_ = self.criterion["pqmf"].synthesis(y_)

            # discriminator loss
            p = self.model["discriminator"](y)
            p_ = self.model["discriminator"](y_.detach())
            real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
            dis_loss = real_loss + fake_loss
            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"],
                )
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        vqidx, aux, mel, prompt, y, xlens, prompt_lens = batch
        vqidx = vqidx.to(self.device).long()
        mel = mel.to(self.device)
        prompt = prompt.to(self.device)
        vqvec = idx2vec(self.feat_codebook, vqidx, self.feat_codebook_numgroups)
        y = y.unsqueeze(-2).to(self.device)  # (B, 1, T)

        # build mask
        mask = make_non_pad_mask(xlens).to(self.device)  # (B, L)
        prompt_mask = make_non_pad_mask(prompt_lens).to(self.device)  # (B, L_prompt)

        #######################
        #      Generator      #
        #######################
        mel_, _, y_ = self.model["generator"](vqvec, prompt, mask, prompt_mask)  # (B, L, 80), (B, C, T)

        # reconstruct the signal from multi-band signal
        if self.config["generator_params"]["out_channels"] > 1:
            y_mb_ = y_
            y_ = self.criterion["pqmf"].synthesis(y_mb_)

        # initialize
        gen_loss = 0.0
        aux_loss = 0.0

        # frontend mel prediction loss
        frontend_mel_pred_loss = F.l1_loss(torch.masked_select(mel, mask.unsqueeze(-1)),
                                           torch.masked_select(mel_, mask.unsqueeze(-1)))
        self.total_eval_loss["eval/frontend_mel_pred_loss"] += frontend_mel_pred_loss.item()
        gen_loss += self.config["lambda_frontend_mel_prediction"] * frontend_mel_pred_loss

        # multi-resolution stft loss
        if self.config["use_stft_loss"]:
            sc_loss, mag_loss = self.criterion["stft"](y_, y)
            aux_loss += sc_loss + mag_loss
            self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
            self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()

        # subband multi-resolution stft loss
        if self.config.get("use_subband_stft_loss", False):
            aux_loss *= 0.5  # for balancing with subband stft loss
            y_mb = self.criterion["pqmf"].analysis(y)
            sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
            self.total_eval_loss["eval/sub_spectral_convergence_loss"] += sub_sc_loss.item()
            self.total_eval_loss["eval/sub_log_stft_magnitude_loss"] += sub_mag_loss.item()
            aux_loss += 0.5 * (sub_sc_loss + sub_mag_loss)

        # mel spectrogram loss
        if self.config["use_mel_loss"]:
            mel_loss = self.criterion["mel"](y_, y)
            aux_loss += mel_loss
            self.total_eval_loss["eval/mel_loss"] += mel_loss.item()

        # weighting stft loss
        gen_loss += aux_loss * self.config.get("lambda_aux", 1.0)

        # adversarial loss
        p_ = self.model["discriminator"](y_)
        adv_loss = self.criterion["gen_adv"](p_)
        gen_loss += self.config["lambda_adv"] * adv_loss

        # feature matching loss
        if self.config["use_feat_match_loss"]:
            p = self.model["discriminator"](y)
            fm_loss = self.criterion["feat_match"](p_, p)
            self.total_eval_loss["eval/feature_matching_loss"] += fm_loss.item()
            gen_loss += (
                    self.config["lambda_adv"] * self.config["lambda_feat_match"] * fm_loss
            )

        #######################
        #    Discriminator    #
        #######################
        p = self.model["discriminator"](y)
        p_ = self.model["discriminator"](y_)

        # discriminator loss
        real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
        dis_loss = real_loss + fake_loss

        # add to total eval loss
        self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        self.total_eval_loss["eval/real_loss"] += real_loss.item()
        self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
        self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.data_loader["dev"], desc="[eval]"), 1):
            # eval one step
            self._eval_step(batch)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(os.path.join(self.config["outdir"], 
                                              f"checkpoint-{self.steps}steps.pkl"))
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class Collator(object):
    """Customized collator for Pytorch DataLoader in training."""

    def __init__(
            self,
            hop_size=256,
            win_length=1024,
            sampling_rate=16000,
            prompt_dim=1024,
            prompt_fold_by_2=False
    ):
        """Initialize customized collator for PyTorch DataLoader.

        Args:
            hop_size (int): Hop size of features, in sampling points.
            win_length (int): window length of features.
            sampling_rate (int): sampling rate of waveform data
            prompt_dim (int): number of prompt embedding dimensions
        """
        self.hop_size = hop_size
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.prompt_dim = prompt_dim
        if prompt_fold_by_2:
            self.prompt_len_factor = 2
        else:
            self.prompt_len_factor = 1

    def construct_prompt(self, mel_lens):
        prompt_lens = [random.randint(int(l / (3 * self.prompt_len_factor)), int(l / (2 * self.prompt_len_factor))) for l in mel_lens]
        prompt_starts = []
        is_from_start = []
        for ml, pl in zip(mel_lens, prompt_lens):
            if random.random() > 0.5:
                # from start
                prompt_start = random.randint(0, 1 * self.sampling_rate // (self.hop_size * self.prompt_len_factor))
                is_from_start.append(True)
            else:
                # from ending
                prompt_start = random.randint((ml - 1 * self.sampling_rate // self.hop_size) // self.prompt_len_factor, ml // self.prompt_len_factor) - pl
                is_from_start.append(False)
            prompt_starts.append(prompt_start)
        return prompt_lens, prompt_starts, is_from_start

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        This collator will automatically determine the prompt segment (acoustic context) for each utterance.
        The prompt is cut off from the current utterance, ranging from one third to half of the original utterance.
        The prompt can be cut from either the starting or the ending of the utterance, within 1 second margin.
        The other features include 3-dim auxiliary features, 2-dim VQ features (2 for number of groups), and D-dim prompts (e.g. WavLM features)

        Returns:
            Tensor ys: waveform batch (B, T).
            Tensors vqs, mels: Auxiliary feature batch (B, C, T'), where T' = T / hop_size.
            Tensor prompts: prompt feature batch (B, C, T'')
            List c_lengths, prompt_lengths: list of lengths
        """
        batch = batch[0]

        # check length
        batch = [self._adjust_length(*b) for b in batch]
        ys, vqs, mels, prompts_old = list(map(list, zip(*batch)))  # [(a,b), (c,d)] -> [a, c], [b, d]

        batch_size = len(vqs)

        prompt_lengths, prompt_starts, is_from_starts = self.construct_prompt([len(m) for m in mels])
        c_lengths = []
        prompts = torch.zeros(batch_size, max(prompt_lengths), self.prompt_dim)
        for i in range(batch_size):
            prompts[i, :prompt_lengths[i]] = torch.tensor(prompts_old[i][prompt_starts[i]:prompt_starts[i]+prompt_lengths[i], :])
            if is_from_starts[i]:
                start_idx = (prompt_starts[i] + prompt_lengths[i])*self.prompt_len_factor
                mels[i] = mels[i][start_idx:]
                vqs[i] = vqs[i][start_idx:]
                ys[i] = ys[i][start_idx * self.hop_size: ]
            else:
                end_idx = prompt_starts[i]*self.prompt_len_factor
                mels[i] = mels[i][:end_idx]
                vqs[i] = vqs[i][:end_idx]
                ys[i] = ys[i][:end_idx * self.hop_size]
            c_lengths.append(len(mels[i]))

        vqs = pad_list([torch.tensor(c) for c in vqs], pad_value=0) # (B, L, Groups)
        vqs = vqs.long()
        mels = pad_list([torch.tensor(c) for c in mels], pad_value=0)  # (B, L, 80)

        ys = pad_list([torch.tensor(y, dtype=torch.float) for y in ys], pad_value=0)[:, :mels.size(1) * self.hop_size]  # (B, T)
        assert ys.size(1) == mels.size(1) * self.hop_size == vqs.size(1) * self.hop_size

        return vqs, mels, prompts, ys, c_lengths, prompt_lengths

    def _adjust_length(self, x, c, *args):
        """Adjust the audio and feature lengths.

        Note:
            Basically we assume that the length of x and c are adjusted
            through preprocessing stage, but if we use other library processed
            features, this process will be needed.

        """
        if len(x) > len(c) * self.hop_size:
            x = x[(self.win_length - self.hop_size) // 2:]
            x = x[:len(c) * self.hop_size]

        # check the legnth is valid
        assert len(x) == len(c) * self.hop_size

        return x, c, *args


def main(rank, n_gpus):
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train vec2wav2 (See detail in vec2wav2/bin/train.py)."
    )
    parser.add_argument(
        "--train-wav-scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file for training. "
    )
    parser.add_argument(
        "--train-vqidx-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for training. "
    )
    parser.add_argument(
        "--train-mel-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for training. "
    )
    parser.add_argument(
        "--train-prompt-scp",
        default=None,
        type=str,
        help="prompt scp (in this case, utt to path)"
    )
    parser.add_argument(
        "--train-aux-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for training. "
    )
    parser.add_argument(
        "--train-segments",
        default=None,
        type=str,
        help="kaldi-style segments file for training.",
    )
    parser.add_argument(
        "--train-num-frames",
        default=None,
        type=str,
        help="kaldi-style utt2num_frames file for training.",
    )
    parser.add_argument(
        "--dev-wav-scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file for validation. "
    )
    parser.add_argument(
        "--dev-vqidx-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for vaidation. "
    )
    parser.add_argument(
        "--dev-mel-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for vaidation. "
    )
    parser.add_argument(
        "--dev-prompt-scp",
        default=None,
        type=str,
        help="prompt scp (in this case, utt to path)"
    )
    parser.add_argument(
        "--dev-aux-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for vaidation. "
    )
    parser.add_argument(
        "--dev-segments",
        default=None,
        type=str,
        help="kaldi-style segments file for validation.",
    )
    parser.add_argument(
        "--dev-num-frames",
        default=None,
        type=str,
        help="kaldi-style utt2num_frames file for validation.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--pretrain",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to load pretrained params. (default="")',
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument("--vq-codebook", default=None, type=str)
    # parser.add_argument("--sampling-rate", type=int)
    # parser.add_argument("--num-mels", type=int)
    # parser.add_argument("--hop-size", type=int)
    # parser.add_argument("--win-length", type=int)
    args = parser.parse_args()

    # init distributed training
    device = torch.device("cuda")
    # effective when using fixed size inputs
    # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.benchmark = True
    # setup for distributed training
    # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
    if n_gpus == 1:
        assert rank == 0

    set_loglevel(args.verbose)
    
    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # init process group
    logging.info("Synchronizing between all workers.")
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=n_gpus, rank=rank)
    torch.cuda.set_device(rank)
    logging.info("Finished init process group.")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config['rank'] = rank
    config['distributed'] = True
    config['world_size'] = n_gpus
    config["version"] = vec2wav2.__version__  # add version info
    if rank == 0:
        with open(os.path.join(args.outdir, "config.yml"), "w") as f:
            yaml.dump(config, f, Dumper=yaml.Dumper)
        for key, value in config.items():
            logging.info(f"{key} = {value}")

    # get dataset
    train_dataset = AudioMelSCPDataset(
        wav_scp=args.train_wav_scp,
        vqidx_scp=args.train_vqidx_scp,
        mel_scp=args.train_mel_scp,
        prompt_scp=args.train_prompt_scp,
        aux_scp=args.train_aux_scp,
        utt2num_frames=args.train_num_frames,
        segments=args.train_segments,
        batch_frames=config.get("batch_frames", None),
        batch_size=config.get("batch_size", None),
        min_num_frames=config.get("min_num_frames", None),
        max_num_frames=config.get("max_num_frames", None),
        allow_cache=config.get("allow_cache", False),  # keep compatibility
        length_tolerance=config.get("length_tolerance", 2),
        prompt_fold_by_2=config.get("prompt_fold_by_2", True)
    )
    if rank == 0:
        logging.info(f"The number of training batches = {len(train_dataset)}.")
    dev_dataset = AudioMelSCPDataset(
        wav_scp=args.dev_wav_scp,
        vqidx_scp=args.dev_vqidx_scp,
        mel_scp=args.dev_mel_scp,
        prompt_scp=args.dev_prompt_scp,
        aux_scp=args.dev_aux_scp,
        utt2num_frames=args.dev_num_frames,
        segments=args.dev_segments,
        min_num_frames=config.get("min_num_frames", None),
        max_num_frames=config.get("max_num_frames", None),
        allow_cache=config.get("allow_cache", False),  # keep compatibility
        length_tolerance=config.get("length_tolerance", 2),
        prompt_fold_by_2=config.get("prompt_fold_by_2", True)
    )
    if rank == 0:
        logging.info(f"The number of development batches = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    collator = Collator(
        hop_size=config["hop_size"],
        win_length=config["win_length"],
        sampling_rate=config["sampling_rate"],
        prompt_dim=config['frontend_params']['prompt_channels'],
        prompt_fold_by_2=config.get("prompt_fold_by_2", True)
    )

    sampler = {
        "train": DistributedSampler(
            dataset=dataset["train"],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        ),
        "dev": DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=False,
        )}
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False,
            collate_fn=collator,
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False,
            collate_fn=collator,
            num_workers=config["num_workers"],
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define models
    generator_class = getattr(
        vec2wav2.models,
        # keep compatibility
        config.get("generator_type", "ParallelWaveGANGenerator"),
    )
    discriminator_class = getattr(
        vec2wav2.models,
        # keep compatibility
        config.get("discriminator_type", "ParallelWaveGANDiscriminator"),
    )
    model = {
        "generator": vec2wav2.models.VEC2WAV2Generator(
            vec2wav2.models.CTXVEC2WAVFrontend(config["prompt_net_type"], config["num_mels"], **config["frontend_params"]),
            generator_class(**config["generator_params"])
        ).to(device),
        "discriminator": discriminator_class(
            **config["discriminator_params"],
        ).to(device),
    }

    # define criteria
    criterion = {
        "gen_adv": GeneratorAdversarialLoss(
            # keep compatibility
            **config.get("generator_adv_loss_params", {})
        ).to(device),
        "dis_adv": DiscriminatorAdversarialLoss(
            # keep compatibility
            **config.get("discriminator_adv_loss_params", {})
        ).to(device),
    }
    if config.get("use_stft_loss", True):  # keep compatibility
        config["use_stft_loss"] = True
        criterion["stft"] = MultiResolutionSTFTLoss(**config["stft_loss_params"]).to(device)
    if config.get("use_subband_stft_loss", False):  # keep compatibility
        assert config["generator_params"]["out_channels"] > 1
        criterion["sub_stft"] = MultiResolutionSTFTLoss(**config["subband_stft_loss_params"]).to(device)
    else:
        config["use_subband_stft_loss"] = False
    if config.get("use_feat_match_loss", False):  # keep compatibility
        criterion["feat_match"] = FeatureMatchLoss(
            # keep compatibility
            **config.get("feat_match_loss_params", {}),
        ).to(device)
    else:
        config["use_feat_match_loss"] = False
    if config.get("use_mel_loss", False):  # keep compatibility
        criterion["mel"] = MelSpectrogramLoss(**config["mel_loss_params"],).to(device)
    else:
        config["use_mel_loss"] = False

    # define optimizers and schedulers
    generator_optimizer_class = getattr(
        vec2wav2.optimizers,
        # keep compatibility
        config.get("generator_optimizer_type", "RAdam"),
    )
    discriminator_optimizer_class = getattr(
        vec2wav2.optimizers,
        # keep compatibility
        config.get("discriminator_optimizer_type", "RAdam"),
    )
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(),
            **config["generator_optimizer_params"],
        ),
        "discriminator": discriminator_optimizer_class(
            model["discriminator"].parameters(),
            **config["discriminator_optimizer_params"],
        ),
    }
    generator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("generator_scheduler_type", "StepLR"),
    )
    discriminator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("discriminator_scheduler_type", "StepLR"),
    )
    scheduler = {
        "generator": generator_scheduler_class(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"],
        ),
        "discriminator": discriminator_scheduler_class(
            optimizer=optimizer["discriminator"],
            **config["discriminator_scheduler_params"],
        ),
    }
    from torch.nn.parallel import DistributedDataParallel
    model["generator"] = DistributedDataParallel(model["generator"], device_ids=[rank], find_unused_parameters=True)
    model["discriminator"] = DistributedDataParallel(model["discriminator"], device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        # show settings
        logging.info(model["generator"])
        logging.info(f"Generator has nparams: {sum([p.numel() for p in model['generator'].parameters()])}")
        logging.info(model["discriminator"])
        logging.info(f"Discriminator has nparams: {sum([p.numel() for p in model['discriminator'].parameters()])}")
        logging.info(optimizer["generator"])
        logging.info(optimizer["discriminator"])

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        if rank == 0:
            logging.info(f"Successfully load parameters from {args.pretrain}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        if rank == 0:
            logging.info(f"Successfully resumed from {args.resume}.")

    # run training loop
    try:
        trainer.run()
    finally:
        if rank == 0:
            trainer.save_checkpoint(os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl"))
            logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CPU training is not allowed."
    n_gpus = torch.cuda.device_count()
    print(f"============> using {n_gpus} GPUS")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8000"

    mp.spawn(
        main,
        nprocs=n_gpus,
        args=(n_gpus,)
    )
