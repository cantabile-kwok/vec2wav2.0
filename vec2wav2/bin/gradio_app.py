#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gradio as gr
import logging
import yaml
import soundfile as sf
import os
from pathlib import Path
from vec2wav2.bin.vc import VoiceConverter, configure_logging, vc_args

# Create Gradio interface
def create_interface():
    args = vc_args()
    logger = configure_logging(args.verbose)
    voice_converter = VoiceConverter(
        expdir=args.expdir,
        token_extractor=args.token_extractor,
        prompt_extractor=args.prompt_extractor,
        prompt_output_layer=args.prompt_output_layer,
        checkpoint=args.checkpoint, 
        script_logger=logger
    )
    with gr.Blocks(title="Voice Conversion") as demo:
        gr.Markdown("# vec2wav 2.0 Voice Conversion Demo")
        gr.Markdown("Upload source audio and target speaker audio to convert the voice.")
        
        with gr.Row():
            source_audio = gr.Audio(label="Source Audio", type="filepath")
            target_audio = gr.Audio(label="Target Speaker Audio", type="filepath")
        
        examples = [
            ["examples/Zuckerberg.wav", "examples/Rachel.wav"],
            ["examples/TheresaMay.wav", "examples/OptimusPrime.wav"]
        ]
        gr.Examples(examples, label="Examples", inputs=[source_audio, target_audio])

        convert_btn = gr.Button("Convert Voice")
        output_audio = gr.Audio(label="Converted Audio")

        convert_btn.click(
            fn=voice_converter.voice_conversion,
            inputs=[source_audio, target_audio],
            outputs=output_audio
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
