# Image Captioning with Visual Attention (Flickr8k)

An encoder-decoder image captioner in TensorFlow/Keras. The decoder attends over a 7×7 grid of image regions for every word it generates.

| Component | Choice |
|---|---|
| Encoder | Frozen EfficientNetB0 (ImageNet). Its 7×7×1280 map is kept as 49 region vectors, with learned 2D position embeddings |
| Decoder | 2-layer Transformer decoder: causal self-attention, cross-attention over the image regions, and a feed-forward block |
| Training | Teacher forcing; token-normalised masked cross-entropy; AdamW with warmup and cosine decay; early stopping on validation loss |
| Splits | 6,000 / 1,000 / 1,091 images (split by image; vocabulary from training captions only) |
| Decoding | Greedy and beam search (k=3, length-normalised) |
| Evaluation | Corpus BLEU-1 to BLEU-4 on held-out test images against all 5 references |

## Results (test set, 1,091 images)

| Decoding | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| Greedy | – | – | – | – |
| Beam (k=3) | – | – | – | – |

*Fill in from `test_bleu.json` after running the notebook.*

The notebook also plots per-word attention maps showing which image regions the model looks at for each word.

## Run

Open `image-captioning-attention.ipynb` on Kaggle, add the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k), enable a GPU, and run all cells.

## Changes from the first version

`image-caption-generation-encoder-decoder.ipynb` was the first version: pooled Xception features with a BiLSTM decoder. The new notebook fixes several issues, listed in its "What changed" section:
- The attention block collapsed to a rank-1 context and never attended to image regions.
- BLEU counted `startseq` / `endseq` tokens.
- Caption cleaning was a no-op.
- Test captions leaked into the word embeddings.
- There was no validation split.
