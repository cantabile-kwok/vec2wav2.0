# Copyright 2024 Yiwei Guo
#  Licensed under Apache 2.0

"""Extract VQ indexes using vq-wav2vec model (from fairseq)"""

import torch
import logging
from kaldiio import WriteHelper
import os
import fairseq
import argparse
import numpy as np
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
from vec2wav2.utils.utils import read_wav_16k

logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

class Extractor:
    def __init__(self, checkpoint="pretrained/vq-wav2vec_kmeans.pt", device="cuda"):
        self.device = device
        self.model, self.cfg, self.task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint])
        self.model = self.model[0].to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
    
    def extract(self, wav: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            audio = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)

            z = self.model.feature_extractor(audio)
            _, idxs = self.model.vector_quantizer.forward_idx(z)
        return idxs[0].cpu()  # [L, Groups]

    def get_codebook(self) -> np.ndarray:
        quantizer = self.model.vector_quantizer
        if self.cfg.model.vq_type == "kmeans":
            codebook = quantizer.expand_embedding.data.transpose(0,1).contiguous()
        elif self.cfg.model.vq_type == "gumbel":
            codebook = quantizer.vars.data
            if quantizer.combine_groups:
                codebook = codebook.repeat(1, quantizer.groups, 1)
            codebook = codebook.view(quantizer.groups, quantizer.num_vars, -1) 

        codebook = codebook.cpu().numpy()
        return codebook

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-scp', type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument('--model', default="pretrained/vq-wav2vec_kmeans.pt", type=str)
    args = parser.parse_args()
    
    extractor = Extractor(checkpoint=args.model, device="cuda" if torch.cuda.is_available() else "cpu")

    out_dir=Path(args.out_dir).absolute()
    with open(args.wav_scp, 'r') as f, torch.no_grad(), WriteHelper(f"ark,scp:{out_dir}/feats.ark,{out_dir}/feats.scp") as writer:
        for line in tqdm(f.readlines()):
            uttid, wav_path = line.strip().split(maxsplit=1)
            logging.info("Extracting " + uttid)
            audio = read_wav_16k(wav_path)
            idxs = extractor.extract(audio).cpu().numpy()
            idxs = idxs.astype(float)
            writer(uttid, idxs)
