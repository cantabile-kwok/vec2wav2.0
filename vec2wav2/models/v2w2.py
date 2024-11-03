# -*- coding: utf-8 -*-
# Copyright 2024 Yiwei Guo
#   Licensed under the Apache 2.0 license.

"""vec2wav2.0 main architectures"""

import torch
from vec2wav2.models.conformer.decoder import Decoder as ConformerDecoder
from vec2wav2.utils import crop_seq
from vec2wav2.models.bigvgan import BigVGAN
from vec2wav2.models.prompt_prenet import ConvPromptPrenet
import logging


class CTXVEC2WAVFrontend(torch.nn.Module):

    def __init__(self,
                 prompt_net_type,
                 num_mels,
                 vqvec_channels,
                 prompt_channels,
                 conformer_params):

        super(CTXVEC2WAVFrontend, self).__init__()

        if prompt_net_type == "ConvPromptPrenet":
            self.prompt_prenet = ConvPromptPrenet(
                embed=prompt_channels,
                conv_layers=[(128, 3, 1, 1), (256, 5, 1, 2), (512, 5, 1, 2), (conformer_params["attention_dim"], 3, 1, 1)],
                dropout=0.1,
                skip_connections=True,
                residual_scale=0.25,
                non_affine_group_norm=False,
                conv_bias=True,
                activation=torch.nn.ReLU()
            )
        elif prompt_net_type == "Conv1d":
            self.prompt_prenet = torch.nn.Conv1d(prompt_channels, conformer_params["attention_dim"], kernel_size=5, padding=2)
        else:
            raise NotImplementedError

        self.encoder1 = ConformerDecoder(vqvec_channels, input_layer='linear', **conformer_params)

        self.hidden_proj = torch.nn.Linear(conformer_params["attention_dim"], conformer_params["attention_dim"])

        self.encoder2 = ConformerDecoder(0, input_layer=None, **conformer_params)
        self.mel_proj = torch.nn.Linear(conformer_params["attention_dim"], num_mels)

    def forward(self, vqvec, prompt, mask=None, prompt_mask=None):
        """
        params:
            vqvec: sequence of VQ-vectors.
            prompt: sequence of mel-spectrogram prompt (acoustic context)
            mask: mask of the vqvec. True or 1 stands for valid values.
            prompt_mask: mask of the prompt.
        vqvec and prompt are of shape [B, D, T]. All masks are of shape [B, T].
        returns:
            enc_out: the input to the vec2wav2 Generator (BigVGAN);
            mel: the frontend predicted mel spectrogram (for faster convergence);
        """
        prompt = self.prompt_prenet(prompt.transpose(1, 2)).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(-2)
        if prompt_mask is not None:
            prompt_mask = prompt_mask.unsqueeze(-2)
        enc_out, _ = self.encoder1(vqvec, mask, prompt, prompt_mask)

        h = self.hidden_proj(enc_out)

        enc_out, _ = self.encoder2(h, mask, prompt, prompt_mask)
        mel = self.mel_proj(enc_out)  # (B, L, 80)

        return enc_out, mel, None


class VEC2WAV2Generator(torch.nn.Module):

    def __init__(self, frontend: CTXVEC2WAVFrontend, backend: BigVGAN):

        super(VEC2WAV2Generator, self).__init__()
        self.frontend = frontend
        self.backend = backend

    def forward(self, vqvec, prompt, mask=None, prompt_mask=None, crop_len=0, crop_offsets=None):
        """
        :param vqvec: (torch.Tensor) The shape is (B, L, D). Sequence of VQ-vectors.
        :param prompt: (torch.Tensor) The shape is (B, L', 80). Sequence of mel-spectrogram prompt (acoustic context)
        :param mask: (torch.Tensor) The dtype is torch.bool. The shape is (B, L). True or 1 stands for valid values in `vqvec`.
        :param prompt_mask: (torch.Tensor) The dtype is torch.bool. The shape is (B, L'). True or 1 stands for valid values in `prompt`.
        :return: frontend predicted mel spectrogram; reconstructed waveform.
        """
        h, mel, _ = self.frontend(vqvec, prompt, mask=mask, prompt_mask=prompt_mask)  # (B, L, adim), (B, L, 80)
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0)
        h = h.transpose(1, 2)
        if crop_len > 0:
            h = crop_seq(h, crop_offsets, crop_len)
        if prompt_mask is not None:
            prompt_avg = prompt.masked_fill(~prompt_mask.unsqueeze(-1), 0).sum(1) / prompt_mask.sum(1).unsqueeze(-1)
        else:
            prompt_avg = prompt.mean(1)
        wav = self.backend(h, prompt_avg)  # (B, C, T)
        return mel, None, wav

    def inference(self, vqvec, prompt):
        h, mel, _ = self.frontend(vqvec, prompt)
        wav = self.backend(h.transpose(1,2), prompt.mean(1))

        return mel, None, wav

