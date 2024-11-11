#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 Yiwei Guo

""" Run VC inference with trained model """

import vec2wav2
from vec2wav2.ssl_models.vqw2v_extractor import Extractor as VQW2VExtractor
from vec2wav2.ssl_models.wavlm_extractor import Extractor as WavLMExtractor
# from vec2wav2.ssl_models.w2v2_extractor import Extractor as W2V2Extractor
import torch
import logging
import argparse
from vec2wav2.utils.utils import load_model, load_feat_codebook, idx2vec, read_wav_16k
import soundfile as sf
import yaml
import os


def configure_logging(verbose):
    if verbose:
        logging.getLogger("vec2wav2.ssl_models.WavLM").setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger("vec2wav2.ssl_models.WavLM").setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)
        logging.basicConfig(level=logging.ERROR)
    
    script_logger = logging.getLogger("script_logger")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s | %(levelname)s | %(message)s'))
    script_logger.addHandler(handler)
    script_logger.setLevel(logging.INFO)
    script_logger.propagate = False
    return script_logger

def vc_args():
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument("-s", "--source", default="examples/source.wav", type=str, 
                        help="source wav path")
    parser.add_argument("-t", "--target", default="examples/target.wav", type=str,
                        help="target speaker prompt path")
    parser.add_argument("-o", "--output", default="output.wav", type=str,
                        help="path of the output wav file")
    
    # optional arguments
    parser.add_argument("--expdir", default="pretrained/", type=str,
                        help="path to find model checkpoints and configs. Will load expdir/generator.ckpt and expdir/config.yml.")
    parser.add_argument('--checkpoint', default=None, type=str, help="checkpoint path (.pkl). If provided, will override expdir.")
    parser.add_argument("--token-extractor", default="pretrained/vq-wav2vec_kmeans.pt", type=str,
                        help="checkpoint or model flag of input token extractor")
    parser.add_argument("--prompt-extractor", default="pretrained/WavLM-Large.pt", type=str,
                        help="checkpoint or model flag of speaker prompt extractor")
    parser.add_argument("--prompt-output-layer", default=6, type=int,
                        help="output layer when prompt is extracted from WavLM.")
    
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")

    args = parser.parse_args()
    return args


class VoiceConverter:
    def __init__(self, expdir="pretrained/", token_extractor="pretrained/vq-wav2vec_kmeans.pt", 
                 prompt_extractor="pretrained/WavLM-Large.pt", prompt_output_layer=6,
                 checkpoint=None, script_logger=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.script_logger = script_logger
        self.log_if_possible(f"Using device: {self.device}")
        
        # set up token extractor
        self.token_extractor = VQW2VExtractor(checkpoint=token_extractor, device=self.device)
        feat_codebook, feat_codebook_numgroups = load_feat_codebook(self.token_extractor.get_codebook(), self.device)
        self.feat_codebook = feat_codebook
        self.feat_codebook_numgroups = feat_codebook_numgroups
        self.log_if_possible(f"Successfully set up token extractor from {token_extractor}")
        
        # set up prompt extractor
        self.prompt_extractor = WavLMExtractor(prompt_extractor, device=self.device, output_layer=prompt_output_layer)
        self.log_if_possible(f"Successfully set up prompt extractor from {prompt_extractor}")
        
        # load VC model
        self.config_path = os.path.join(expdir, "config.yml")
        with open(self.config_path) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)
        if checkpoint is not None:
            checkpoint = os.path.join(expdir, checkpoint)
        else:
            checkpoint = os.path.join(expdir, "generator.ckpt")
        self.model = load_model(checkpoint, self.config)
        self.log_if_possible(f"Successfully set up VC model from {checkpoint}")
        
        self.model.backend.remove_weight_norm()
        self.model.eval().to(self.device)
        
    @torch.no_grad()
    def voice_conversion(self, source_audio, target_audio, output_path="output.wav"):
        self.log_if_possible(f"Performing VC from {source_audio} to {target_audio}")
        source_wav = read_wav_16k(source_audio)
        target_wav = read_wav_16k(target_audio)
        vq_idx = self.token_extractor.extract(source_wav).long().to(self.device)
        
        vqvec = idx2vec(self.feat_codebook, vq_idx, self.feat_codebook_numgroups).unsqueeze(0)
        prompt = self.prompt_extractor.extract(target_wav).unsqueeze(0).to(self.device)
        converted = self.model.inference(vqvec, prompt)[-1].view(-1)
        sf.write(output_path, converted.cpu().numpy(), self.config['sampling_rate'])
        self.log_if_possible(f"Saved audio file to {output_path}")
        return output_path
    
    def log_if_possible(self, msg):
        if self.script_logger is not None:
            self.script_logger.info(msg)


if __name__ == "__main__":
    args = vc_args()
    script_logger = configure_logging(args.verbose)
    
    source_wav = read_wav_16k(args.source)
    target_prompt = read_wav_16k(args.target)
    
    with torch.no_grad():
        voice_converter = VoiceConverter(expdir=args.expdir, token_extractor=args.token_extractor,
                                         prompt_extractor=args.prompt_extractor, prompt_output_layer=args.prompt_output_layer,
                                         checkpoint=args.checkpoint, script_logger=script_logger)
        voice_converter.voice_conversion(args.source, args.target, args.output)
