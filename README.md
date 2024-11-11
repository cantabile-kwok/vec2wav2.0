# vec2wav 2.0: Advancing Voice Conversion via Discrete Token Vocoders

[![paper](https://img.shields.io/badge/paper-arxiv:2409.01995-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2409.01995)
[![demo](https://img.shields.io/badge/demo-page-green)](https://cantabile-kwok.github.io/vec2wav2/)
![docker](https://img.shields.io/badge/Docker-blue?logo=docker&logoColor=white)
![python](https://img.shields.io/badge/Python_3.10-orange?logo=python&logoColor=white)

> [!IMPORTANT] 
> Pretrained model (with vq-wav2vec as input) and training procedure are released!

## Environment

Please refer to `environment` directory for a `requirements.txt` and `Dockerfile`.

In addition, for convenience, we also provide a Docker image for Linux, so you can easily run the Docker container:

```sh
docker pull cantabilekwok511/vec2wav2.0:v0.2
docker run -it -v /path/to/vec2wav2.0:/workspace cantabilekwok511/vec2wav2.0:v0.2
```

## Voice Conversion with Pretrained Model
We provide a simple VC interface.

First, please make sure some required models are downloaded in the `pretrained/` directory:

1. vq-wav2vec model from [this url](https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt)
2. WavLM-Large from [this url](https://github.com/microsoft/unilm/blob/master/wavlm/README.md)
3. Pre-trained vec2wav 2.0 (on vq-wav2vec tokens) from [🤗Huggingface](https://huggingface.co/cantabile-kwok/vec2wav2.0/tree/main)

The resulting directory should look like this:
```
pretrained/
    - vq-wav2vec_kmeans.pt 
    - WavLM-Large.pt 
    - generator.ckpt
    - config.yml
```

Then VC can be done by
```
source path.sh
vc.py -s $source_wav -t $speaker_prompt -o $output_wav
```
where `$source_wav, $speaker_prompt` should both be mono-channel audio and preferably `.wav` files.
This script by default tries to load `pretrained/generator.ckpt` and the corresponding `config.yml`. You can provide `--expdir` to change this path.

If you have trained you own model under $expdir, please specify the checkpoint filename:
```
vc.py -s $source_wav -t $speaker_prompt -o $output_wav \
      --expdir $expdir --checkpoint /path/to/checkpoint.pkl
```

### Web Interface

We also provide a VC web interface using Gradio. To launch it:

```
# Make sure gradio is installed first
pip install gradio
python vec2wav2/bin/gradio_app.py
```

This will start a local web server and open the interface in your browser. You can:
1. Upload source audio (the voice you want to convert)
2. Upload target speaker audio (the voice you want to convert to)
3. Click "Convert Voice" to perform the conversion
4. Listen to or download the converted audio

The web interface uses the same models and settings as the command-line tool.


## Training
First, we need to set up data manifests and features. Please refer to [`./data_prep.md`](./data_prep.md) for a guide on LibriTTS dataset.

Then, please refer to [`./train.sh`](./train.sh) for training. It will automatically launch pytorch DDP training on all the devices in `CUDA_VISIBLE_DEVICES`. Please change `os.environ["MASTER_PORT"]` in `vec2wav2/bin/train.py` if you need.

## Decoding (VQ tokens to wav)

If you want to decode VQ features in existing `feats.scp` into wavs, you can use
```
decode.py --feats-scp /path/to/feats.scp --prompt-scp /path/to/prompt.scp \
          --checkpoint /path/to/checkpoint.pkl --config /path/to/config.yml \
          --outdir /path/to/output_dir
```
Here, `prompt.scp` specifies every utterance (content VQ tokens) and its prompts (WavLM features). It is organized in a similar style with `feats.scp`.

## Citation
```
@article{guo2024vec2wav,
  title={vec2wav 2.0: Advancing Voice Conversion via Discrete Token Vocoders},
  author={Guo, Yiwei and Li, Zhihan and Li, Junjie and Du, Chenpeng and Wang, Hankun and Wang, Shuai and Chen, Xie and Yu, Kai},
  journal={arXiv preprint arXiv:2409.01995},
  year={2024}
}
```

## 🔍 See Also: The vec2wav family
<!-- As the name implies, "vec" means code-vectors (with speech discrete tokens), and "wav" means the corresponding wavforms.  -->
<!-- The vec2wav family are speech token vocoders that are important modules in speech generation based on discrete tokens (esp. semantic tokens!). -->

* [[paper]](https://arxiv.org/abs/2204.00768) **vec2wav** in VQTTS. Single-speaker.
* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29747)[[code]](https://github.com/X-LANCE/UniCATS-CTX-vec2wav) **CTX-vec2wav** in UniCATS. Multi-speaker with acoustic prompts. Lots of code borrowed from there.
* 🌟(This) **vec2wav 2.0**. Enhanced in timbre controllability, best for VC!

## Acknowledgements

* [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) for the whole project structure.
* [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN) for the vocoder backbone.
* [Kaldi](https://github.com/kaldi-asr/kaldi) and [ESPnet](https://github.com/espnet/espnet) for providing useful tools and Conformer implementation.
* [Fairseq](https://github.com/facebookresearch/fairseq) for some network architectures.
