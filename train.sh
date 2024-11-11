#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# Modified by Yiwei Guo (2024)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=1              # stage to start
stop_stage=1        # stage to stop
verbose=1             # verbosity level (lower is less info)

conf=conf/vec2wav2.v1.yaml

# dataset setting
part="all"

# directory path setting
datadir=$PWD/data
featdir=$PWD/feats

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)

train_set="train_${part}"       # name of training data directory
dev_set="dev_${part}"           # name of development data directory
eval_set="eval_${part}"         # name of evaluation data directory

# shellcheck disable=SC1091
. parse_options.sh || exit 1;

set -eo pipefail

if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_$(basename "${conf}" .yaml)"
else
    expdir="exp/${train_set}_${tag}"
fi

mkdir -p $expdir

# if $resume is not specified, automatically resumes from the last checkpoint
last_checkpoint=""
if compgen -G "${expdir}/*.pkl" > /dev/null; then
    last_checkpoint=$(ls -dt "${expdir}"/*.pkl | head -1)
fi

if [ -z $resume ]; then
    resume=$last_checkpoint
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    echo "Hostname: `hostname`."
    echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu 1 "${expdir}/log/train.log" \
        train.py \
            --config "${conf}" \
            --train-wav-scp $datadir/${train_set}/wav.scp \
            --train-vqidx-scp ${featdir}/vqidx/${train_set}/feats.scp \
            --train-mel-scp ${featdir}/normed_fbank/${train_set}/feats.scp \
            --train-prompt-scp $featdir/wavlm_l6/$train_set/feats.scp \
            --train-num-frames ${datadir}/${train_set}/utt2num_frames \
            --dev-wav-scp ${datadir}/${dev_set}/wav.scp \
            --dev-vqidx-scp ${featdir}/vqidx/${dev_set}/feats.scp \
            --dev-mel-scp ${featdir}/normed_fbank/${dev_set}/feats.scp \
            --dev-prompt-scp $featdir/wavlm_l6/$dev_set/feats.scp \
            --dev-num-frames $datadir/${dev_set}/utt2num_frames \
            --vq-codebook $featdir/vqidx/codebook.npy \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}"
    echo "Successfully finished training."
fi
