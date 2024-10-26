import torch
from vec2wav2.ssl_models.WavLM import WavLM, WavLMConfig
import soundfile as sf
from vec2wav2.utils.espnet_utils import pad_list, make_pad_mask
import time


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
    wav_path_list = [
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000003_000000.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000004_000000.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000005_000000.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000005_000001.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000005_000001.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000005_000001.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000005_000001.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000005_000001.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000005_000001.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000005_000001.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000005_000001.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000005_000001.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000005_000001.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000005_000001.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000005_000001.wav",
        "dataset/LibriTTS/16k-dev-other/116/288045/116_288045_000008_000004.wav"
    ]*10
    wav_list = []
    for path in wav_path_list:
        audio, fs = sf.read(path)
        wav_list.append(audio)
    wav_lens = [len(x) for x in wav_list]
    conv_kernels = [10, 3, 3, 3, 3, 2, 2]
    conv_strides = [5, 2, 2, 2, 2, 2, 2]
    out_lens = []
    for L in wav_lens:
        x = L
        for k, s in zip(conv_kernels, conv_strides):
            x = calc_out_len(x, k, s)
        out_lens.append(x)
    print("wav lens:", wav_lens)
    print('calculated output lens:', out_lens)
    # wav_tensor = torch.zeros(size=(len(wav_path_list), max(wav_lens)))
    # for i, wav in enumerate(wav_list):
    #     wav_tensor[i, :len(wav)] = torch.from_numpy(wav).float()
    # print(wav_tensor)
    print('begins batch mode')
    extractor = Extractor(checkpoint="pretrained/WavLM-Large.pt")
    s = time.time()
    feat = extractor.extract_batch(wav_list, out_lens)
    t = time.time()
    # feat = extractor.extract(audio)
    # print(feat.shape)
    # for i in range(len(feat)):
    #     print(feat[i, :out_lens[i]])
    print('batch mode cost', t-s, 's')

    print('begins single mode')
    # single mode
    s = time.time()
    for wav in wav_list:
        feat = extractor.extract(wav)
        # print(feat)
    t = time.time()
    print('single mode cost', t-s, 's')


