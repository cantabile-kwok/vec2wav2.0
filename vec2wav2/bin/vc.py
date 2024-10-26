#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import vec2wav2
from vec2wav2.ssl_models.vqw2v_extractor import Extractor as VQW2VExtractor
from vec2wav2.ssl_models.wavlm_extractor import Extractor as WavLMExtractor
from vec2wav2.ssl_models.w2v2_extractor import Extractor as W2V2Extractor
import torch
import logging
import argparse
from vec2wav2.utils.utils import load_model, load_feat_codebook, idx2vec
import soundfile as sf
import torchaudio.transforms as transforms
import yaml


def read_wav_16k(path):
    wav, sr = sf.read(path)
    if sr != 16000:
        audio_tensor = torch.tensor(wav, dtype=torch.float32)
        resampler = transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(audio_tensor)
        wav = wav.numpy()
    return wav

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument("-s", "--source", default="", required=True, type=str, 
                        help="source wav path")
    parser.add_argument("-t", "--target", default="", required=True, type=str,
                        help="target speaker prompt path")
    parser.add_argument("-o", "--output", default="output.wav", type=str,
                        help="path of the output wav file")
    
    # optional arguments
    parser.add_argument("--expdir", default="pretrained/", type=str,
                        help="path to find model checkpoints and configs")
    parser.add_argument("--token-extractor", default="pretrained/vq-wav2vec_kmeans.pt", type=str,
                        help="checkpoint or model flag of input token extractor")
    parser.add_argument("--prompt-extractor", default="pretrained/WavLM-Large.pt", type=str,
                        help="checkpoint or model flag of speaker prompt extractor")
    parser.add_argument("--prompt-output-layer", default=6, type=int,
                        help="output layer when prompt is extracted from WavLM.")
    
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")

    args = parser.parse_args()
    script_logger = configure_logging(args.verbose)
    
    source_wav = read_wav_16k(args.source)
    target_prompt = read_wav_16k(args.target)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_logger.info(f"Using {device}")
    
    with torch.no_grad():
        # load vq token extractor
        vq_extractor = VQW2VExtractor(checkpoint=args.token_extractor, device=device)
        script_logger.info(f"Successfully set up token extractor from {args.token_extractor}")
        codebook = vq_extractor.get_codebook()
        vq_idx = vq_extractor.extract(source_wav)  # [L, Groups]
        vq_idx = vq_idx.long().to(device)  # [L, Groups]

        # load vq codebook
        feat_codebook, feat_codebook_numgroups = load_feat_codebook(codebook, device)
        
        vqvec = idx2vec(feat_codebook, vq_idx, feat_codebook_numgroups).unsqueeze(0)  # (1, L, D)

        # extract prompt
        prompt_extractor = WavLMExtractor(args.prompt_extractor, device=device, output_layer=args.prompt_output_layer)
        script_logger.info(f"Successfully set up prompt extractor from {args.prompt_extractor}")
        prompt = prompt_extractor.extract(target_prompt)
        prompt = prompt.unsqueeze(0).to(device)
        vqvec = vqvec.to(prompt.dtype)

        # load VC model
        with open(f"{args.expdir}/config.yml") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        checkpoint = f"{args.expdir}/generator.ckpt"  # TODO: need think
        model = load_model(checkpoint, config)
        script_logger.info(f"Successfully set up VC model from {checkpoint}")
        model.backend.remove_weight_norm()
        model.eval().to(device)
        
        # perform actual conversion
        script_logger.info("Performing VC...")
        converted = model.inference(vqvec, prompt)[-1].view(-1)

        # save output wav
        sf.write(args.output, converted.cpu().numpy(), config['sampling_rate'])  # TODO: check sampling_rate key 
        script_logger.info(f"Saved audio file to {args.output}")
        
