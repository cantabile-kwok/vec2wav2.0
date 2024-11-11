#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gradio as gr
import torch
import logging
import yaml
import soundfile as sf
import os
from pathlib import Path
import torchaudio.transforms as transforms
from vec2wav2.ssl_models.vqw2v_extractor import Extractor as VQW2VExtractor
from vec2wav2.ssl_models.wavlm_extractor import Extractor as WavLMExtractor
from vec2wav2.utils.utils import load_model, load_feat_codebook, idx2vec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_wav_16k(audio_path):
    """Process audio file to 16kHz sample rate"""
    if isinstance(audio_path, tuple):  # Gradio audio input returns (sample_rate, audio_data)
        wav = audio_path[1]
        sr = audio_path[0]
    else:  # Regular file path
        wav, sr = sf.read(audio_path)
        
    if sr != 16000:
        audio_tensor = torch.tensor(wav, dtype=torch.float32)
        resampler = transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(audio_tensor)
        wav = wav.numpy()
    return wav

def voice_conversion(source_audio, target_audio, 
                    expdir="pretrained/",
                    token_extractor="pretrained/vq-wav2vec_kmeans.pt",
                    prompt_extractor="pretrained/WavLM-Large.pt",
                    prompt_output_layer=6):
    """Perform voice conversion between source and target audio"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Process input audio
    source_wav = read_wav_16k(source_audio)
    target_wav = read_wav_16k(target_audio)
    
    with torch.no_grad():
        # Set up VQ token extractor
        vq_extractor = VQW2VExtractor(checkpoint=token_extractor, device=device)
        logger.info(f"Token extractor loaded from {token_extractor}")
        
        codebook = vq_extractor.get_codebook()
        vq_idx = vq_extractor.extract(source_wav)
        vq_idx = vq_idx.long().to(device)

        # Load VQ codebook
        feat_codebook, feat_codebook_numgroups = load_feat_codebook(codebook, device)
        vqvec = idx2vec(feat_codebook, vq_idx, feat_codebook_numgroups).unsqueeze(0)

        # Extract prompt
        prompt_extractor = WavLMExtractor(prompt_extractor, device=device, 
                                        output_layer=prompt_output_layer)
        logger.info(f"Prompt extractor loaded from {prompt_extractor}")
        
        prompt = prompt_extractor.extract(target_wav)
        prompt = prompt.unsqueeze(0).to(device)
        vqvec = vqvec.to(prompt.dtype)

        # Load VC model
        config_path = os.path.join(expdir, "config.yml")
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
            
        checkpoint = os.path.join(expdir, "generator.ckpt")
        model = load_model(checkpoint, config)
        logger.info(f"VC model loaded from {checkpoint}")
        
        model.backend.remove_weight_norm()
        model.eval().to(device)

        # Perform conversion
        logger.info("Converting voice...")
        converted = model.inference(vqvec, prompt)[-1].view(-1)

        # Save output
        output_path = "output.wav"
        sf.write(output_path, converted.cpu().numpy(), config['sampling_rate'])
        logger.info(f"Saved output to {output_path}")
        
        return output_path

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Voice Conversion") as demo:
        gr.Markdown("# Voice Conversion Demo")
        gr.Markdown("Upload source audio and target speaker audio to convert the voice.")
        
        with gr.Row():
            source_audio = gr.Audio(label="Source Audio", type="filepath")
            target_audio = gr.Audio(label="Target Speaker Audio", type="filepath")
            
        convert_btn = gr.Button("Convert Voice")
        output_audio = gr.Audio(label="Converted Audio")
        
        convert_btn.click(
            fn=voice_conversion,
            inputs=[source_audio, target_audio],
            outputs=output_audio
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
