# Copyright 2024 Yiwei Guo
#  Licensed under Apache 2.0

"""Extract VQ indexes using WavLM model (from microsoft UniLM)"""

import torch
from vec2wav2.ssl_models.WavLM import WavLM, WavLMConfig
import soundfile as sf
from vec2wav2.utils.espnet_utils import pad_list, make_pad_mask
import time
from pathlib import Path
import argparse
from kaldiio import WriteHelper
from tqdm import tqdm
import logging
from vec2wav2.utils.utils import read_wav_16k

class Extractor:
    def __init__(self, checkpoint="pretrained/WavLM-Large.pt", device="cuda", output_layer=6):
        self.device = device
        checkpoint = torch.load(checkpoint)
        self.cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(self.cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.output_layer = output_layer

    def extract(self, wav):
        with torch.no_grad():
            wav_input_16khz = torch.from_numpy(wav).unsqueeze(0).float().to(self.device)
            if self.cfg.normalize:
                wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz, wav_input_16khz.shape)
            rep = self.model.extract_features(wav_input_16khz, output_layer=self.output_layer)[0]
            return rep.squeeze(0).clone().detach()  # torch.tensor [T, D]

    def extract_batch(self, wav_list, frame_lens):
        # suppose wav is already a tensor padded with 0
        # should be careful with LayerNorm since it may cause difference between batch vs single modes.
        pad_mask = make_pad_mask(frame_lens).to(self.device)
        with torch.no_grad():
            wav_input_16khz = [torch.from_numpy(wav).float().to(self.device) for wav in wav_list]
            if self.cfg.normalize:
                wav_input_16khz = [torch.nn.functional.layer_norm(wav, wav.shape) for wav in wav_input_16khz]
            wav_input_16khz = pad_list(wav_input_16khz, 0)
            s = time.time()
            rep = self.model.extract_features(wav_input_16khz, output_layer=self.output_layer, padding_mask=pad_mask)[0]
            t = time.time()
            print(f'in batch mode, pure extracting costs {t-s} s')
            return rep.clone().detach()  # [B, T, D]


def calc_out_len(in_len, k, s):
    return int((in_len-(k-1)-1)/s + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-scp', type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument('--model', default="pretrained/WavLM-Large.pt", type=str)
    parser.add_argument('--output-layer', default=6, type=int)
    args = parser.parse_args()
    
    extractor = Extractor(checkpoint=args.model, 
                          device="cuda" if torch.cuda.is_available() else "cpu", 
                          output_layer=args.output_layer)
    
    out_dir=Path(args.out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.wav_scp, 'r') as f, torch.no_grad(), WriteHelper(f"ark,scp:{out_dir}/feats.ark,{out_dir}/feats.scp") as writer:
        for line in tqdm(f.readlines()):
            uttid, wav_path = line.strip().split(maxsplit=1)
            logging.info("Extracting " + uttid)
            audio = read_wav_16k(wav_path)
            rep = extractor.extract(audio)
            rep = rep.cpu().numpy()
            writer(uttid, rep)
    