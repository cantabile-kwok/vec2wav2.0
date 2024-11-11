#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# Modified by Yiwei Guo, 2024

"""Decode with trained vec2wav Generator."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

from vec2wav2.datasets import MelSCPDataset
from vec2wav2.utils import load_model, load_feat_codebook, idx2vec


def set_loglevel(verbose):
    # set logger
    if verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Decode from audio tokens and acoustic prompts with trained vec2wav model"
        "(See detail in vec2wav2/bin/decode.py)."
    )
    parser.add_argument(
        "--feats-scp",
        "--scp",
        default=None,
        type=str,
        required=True,
        help="kaldi-style feats.scp file. "
    )
    parser.add_argument(
        "--prompt-scp", 
        default=None, 
        type=str,
        help="kaldi-style prompt.scp file. Similar to feats.scp."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="yaml format configuration file. if not explicitly provided, "
        "it will be searched in the checkpoint directory. (default=None)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()
    set_loglevel(args.verbose)
    
    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # get dataset
    dataset = MelSCPDataset(
        vqidx_scp=args.feats_scp,
        prompt_scp=args.prompt_scp,
        return_utt_id=True,
    )
    logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}.")

    model = load_model(args.checkpoint, config)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    
    model.backend.remove_weight_norm()
    model = model.eval().to(device)

    # load vq codebook
    feat_codebook, feat_codebook_numgroups = load_feat_codebook(np.load(config["vq_codebook"], allow_pickle=True), device)

    # start generation
    total_rtf = 0.0
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for idx, batch in enumerate(pbar, 1):
            utt_id, vqidx, prompt = batch[0], batch[1], batch[2]

            vqidx = torch.tensor(vqidx).to(device)  # (L, G)
            prompt = torch.tensor(prompt).unsqueeze(0).to(device)  # (1, L', D')

            vqidx = vqidx.long()
            vqvec = idx2vec(feat_codebook, vqidx, feat_codebook_numgroups).unsqueeze(0)  # (1, L, D)

            # generate
            start = time.time()
            y = model.inference(vqvec, prompt)[-1].view(-1)
            rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            tgt_dir = os.path.dirname(os.path.join(config["outdir"], f"{utt_id}.wav"))
            os.makedirs(tgt_dir, exist_ok=True)
            basename = os.path.basename(f"{utt_id}.wav")
            # save as PCM 16 bit wav file
            sf.write(
                os.path.join(tgt_dir, basename),
                y.cpu().numpy(),
                config["sampling_rate"],
                "PCM_16",
            )

    # report average RTF
    logging.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f}).")


if __name__ == "__main__":
    main()
