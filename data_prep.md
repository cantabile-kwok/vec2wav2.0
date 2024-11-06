# Data Preparation
In this repo, we rely on the Kaldi-style data formatting. 
We take the LibriTTS (including `clean` and `other` partitions) for example.

### `data/` directory: data manifests
We have organized the `data` directory containing all the LibriTTS data. Here are the steps to establish the `data` dir.
1. Please download from [here](https://huggingface.co/datasets/cantabile-kwok/libritts-all-kaldi-data/resolve/main/data_24k.zip?download=true) (about 5MB), and unzip it to `data` in the project root. Every sub-directory contains `utt2spk, spk2utt` and `wav.scp` files. They are all plain texts, with `<key> <value>` in each line.
2. Then, change the paths in `wav.scp` to the correct ones in your machine.

### `feats/` directory: speech features
We include three types of speech features in vec2wav 2.0. They should all be extracted offline and stored in `./feats/`.
* **VQ index (together with codebook) from vq-wav2vec**. We extracted it by [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#vq-wav2vec), 
and we provide the extracted VQ index sequences with codebook online.
  1. Please download from [here](https://huggingface.co/datasets/cantabile-kwok/libritts-all-kaldi-data/resolve/main/vqidx.zip) (460MB; [here](https://www.modelscope.cn/api/v1/datasets/CantabileKwok/libritts-all-kaldi-data/repo?Revision=master&FilePath=vqidx.zip) for Chinese users).
  2. Unzip it to `feats/vqidx`, and change the corresponding paths in the `feats.scp`. 
  3. You can check out the feature shape by `feat-to-shape.py scp:feats/vqidx/eval_all/feats.scp | head`. The shapes should be `(frames, 2)`.
  4. Get the number of frames: 
        ```bash
        for name in train_all dev_all eval_all; do
          feat-to-len.py scp:feats/vqidx/$name/feats.scp > data/$name/utt2num_frames
        done
        ```
   
Note that you can use `vec2wav2/ssl_models/vqw2v_extractor.py` to extract these indexes locally.

* **Mel spectrograms (FBanks)**. As they are too large, we provide a script to extract them locally:
  ```bash
  nj=64  # parallel jobs. Set this according to your CPU cores.
  bash extract_fbank.sh --nj $nj --stage 0 --stop_stage 1  # Default: 80-dim with 10ms frame shift
  # Stage 0 extracts fbank in parallel. Stage 1 performs normalization.
  ```
This will create `feats/fbank` and `feats/normed_fbank` each about 16GB. You can delete `feats/fbank` after normalization (just it would be better if you keep the `train_all/cmvn.ark` there).

* **WavLM features**. As they are too large, please use `vec2wav2/ssl_models/wavlm_extractor.py` to extract them locally:
  ```bash
  name=train_all  # change to dev_all or eval_all for different splits
  python vec2wav2/ssl_models/wavlm_extractor.py --wav-scp data/$name/wav_scp \
                     --out_dir feats/wavlm_l6/$name/ --output-layer 6
  ```
  This will create `feats/wavlm_l6/$name/feats.ark` and `feats.scp`. ⚠️Note that the WavLM features for entire training set can be very large (~380GB)! It is also reasonable to extract them on-the-fly, but this might slow down training.

Finally, you have correctly formatted the data for training!
