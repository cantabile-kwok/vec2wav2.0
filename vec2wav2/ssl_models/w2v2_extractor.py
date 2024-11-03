# Copyright 2024 Yiwei Guo
#  Licensed under Apache 2.0

"""Extract VQ indexes using wav2vec2.0 model (from fairseq)"""

import torch
import logging
from kaldiio import WriteHelper
import os
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining
import argparse
import numpy as np
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

class Extractor:
    def __init__(self, checkpoint="pretrained/wav2vec2-large-lv60/", device="cuda"):
        self.device = device
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint) 
        model = Wav2Vec2ForPreTraining.from_pretrained(checkpoint) 
        model.to(self.device)
        model.half()
        model.eval()
        self.model = model
        self.feature_extractor = feature_extractor
        logging.info(self.model)
        for p in self.model.parameters():
            p.requires_grad_(False)
    
    def extract(self, wav: np.ndarray, sample_rate: int) -> torch.Tensor:
        with torch.no_grad():
            wav = torch.from_numpy(wav).float()

            input_values = self.feature_extractor(wav, return_tensors="pt", sampling_rate=sample_rate).input_values
            input_values = input_values.half().to(self.device)
            outputs = self.model.wav2vec2(input_values)
            extract_features = self.model.dropout_features(outputs[1]) 
            hidden_states = extract_features
            batch_size, sequence_length, hidden_size = hidden_states.shape
            hidden_states = self.model.quantizer.weight_proj(hidden_states)
            hidden_states = hidden_states.view(batch_size * sequence_length * self.model.quantizer.num_groups, -1)
            codevector_idx = hidden_states.argmax(dim=-1)
            idxs = codevector_idx.view(batch_size, sequence_length, self.model.quantizer.num_groups)
        return idxs[0].cpu()  # [L, Groups]

    def get_codebook(self) -> np.ndarray:
        quantizer = self.model.quantizer
        codebook = quantizer.codevectors  # (1, 640, 384)
        codebook = codebook.view(quantizer.num_groups, quantizer.num_vars, -1)  # (2, 320, 384)
        return codebook.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-scp', type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument('--model', default="pretrained/wav2vec2-large-lv60/", type=str)
    args = parser.parse_args()
    
    extractor = Extractor(checkpoint=args.model, device="cuda" if torch.cuda.is_available() else "cpu")

    out_dir=Path(args.out_dir).absolute()
    with open(args.wav_scp, 'r') as f, torch.no_grad(), WriteHelper(f"ark,scp:{out_dir}/feats.ark,{out_dir}/feats.scp") as writer:
        for line in tqdm(f.readlines()):
            uttid, wav_path = line.strip().split(maxsplit=1)
            logging.info("Extracting " + uttid)
            audio, sample_rate = sf.read(wav_path)
            idxs = extractor.extract(audio, sample_rate=sample_rate)
            idxs = idxs.astype(float)
            writer(uttid, idxs)
