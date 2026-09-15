# Automatic Image Captioning

This project generates descriptive captions for images. A Convolutional Neural Network (CNN) reads the image and turns it into a feature vector, and a Recurrent Neural Network (LSTM) reads that vector and writes a sentence one word at a time.

Everything lives in one notebook, `AutomaticImageCaptioning.ipynb`, and there is a small Streamlit app for trying the trained model on your own images.

## Model Architecture

### Encoder

A ResNet50 pretrained on ImageNet, with its classification layer replaced by a linear layer that projects into the embedding space. How much of it trains is configurable:

- `none` keeps the backbone frozen and trains only the projection head
- `layer4` also trains the final residual block
- `all` trains the whole backbone

When any part of the backbone is unfrozen, its BatchNorm layers are held in inference mode. Those layers carry statistics estimated over ImageNet's 1.2 million images, and recomputing them from batches of 32 makes them noisy enough to undo the benefit of fine-tuning.

### Decoder

A single-layer LSTM with an embedding size of 256 and a hidden size of 512, with dropout of 0.5 before the output layer. The image feature is fed in as the first timestep, then `<SOS>`, then the words.

### Vocabulary

Built from the training split only, keeping words that appear at least 5 times. Special tokens are `<PAD>`, `<SOS>`, `<EOS>` and `<UNK>`.

## Dataset

[Flickr 8k](https://www.kaggle.com/datasets/adityajn105/flickr8k): 8,091 images with 5 captions each, so 40,455 caption rows in total.

The split is done on **images**, not on caption rows. Splitting rows would put four captions of an image into training and the fifth into validation, which means the model has already seen every image it is validated on. Splitting by image gives a genuinely held-out set, and each validation image keeps all five of its captions as BLEU references.

| | images | captions |
| --- | --- | --- |
| train | 6,473 | 32,365 |
| validation | 1,618 | 8,090 |

## Image Preprocessing

Images are decoded once and held in RAM as 256x256 uint8 tensors, about 1.6 GB for the whole dataset. Reading JPEGs off disk was taking 437 ms per batch against 21 ms of GPU work, so the GPU was idle roughly 95% of the time. Caching the decoded pixels fixes that while still allowing fresh random crops each epoch and still allowing the CNN to be fine-tuned, neither of which works if you cache the encoder's output instead.

| step | description |
| --- | --- |
| cache | decode once, resize to 256x256, keep in RAM as uint8 |
| training crop | random 224x224 crop plus random horizontal flip |
| validation crop | deterministic centre 224x224 crop |
| normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |

## Training

| setting | value |
| --- | --- |
| epochs | 100, with early stopping after 15 epochs without improvement |
| batch size | 32 |
| optimizer | Adam |
| learning rate | 3e-4 for the decoder, 1e-5 for the CNN once unfrozen |
| scheduler | cosine annealing across the full run |
| gradient clipping | max norm 5.0 |
| loss | cross-entropy, ignoring `<PAD>` |

Fine-tuning happens in two stages. The decoder starts from random weights and produces large, uninformative gradients, so pushing those into a pretrained ResNet damages it before the decoder can benefit. The backbone stays frozen for the first 10 epochs and is unfrozen after that at a much lower learning rate.

## Evaluation

BLEU compares a generated caption against the real ones by counting matching word sequences. We report BLEU-1 through BLEU-4, using `corpus_bleu` with smoothing method 4.

Two things about how it is measured:

- Scores are pooled across the whole validation set with `corpus_bleu` rather than averaged over per-sentence scores. Averaging lets a two-word caption count as much as a twelve-word one, and published Flickr8k numbers use the pooled version, so this makes ours comparable to theirs.
- One caption is generated per image and compared against all five references at once, instead of generating the same caption five times and grading each against a single reference.

BLEU-1 through BLEU-4 all come from the same generated captions, so measuring all four costs nothing extra.

## Experiments

Each configuration changes exactly one thing against the baseline, so any difference in the results can be traced to that change.

| experiment | what changes |
| --- | --- |
| `baseline_frozen` | nothing, this is the reference point |
| `finetune_layer4` | unfreezes ResNet's final residual block after epoch 10 |
| `finetune_all` | unfreezes the whole backbone after epoch 10 |
| `larger_decoder` | LSTM hidden size 512 to 1024 |
| `no_augment` | centre crop only, no random crop or flip |

Beam width is not in this list because it only affects caption generation, never training. It is swept over the finished model instead, which costs a minute rather than a full run.

## Results

Pending. The numbers here were produced before a bug in the training loop was found and fixed, so they are not meaningful and have been removed rather than left to mislead. This section will be refilled once the experiments above have been run.

For reference, the bug: the loss paired each decoder output with the token two places ahead instead of the next one. Training loss fell normally, which made it look healthy, but the model was learning to skip a word. Captions came out as things like "Man a on bike a" instead of "a man on a bike". BLEU-4 sat at 0.0199 for ten epochs while the loss dropped 28%, and that contradiction was the clue.

## Web Application

```bash
streamlit run app.py
```

Upload an image and the app generates a caption with the trained model.

## Project Structure

```
.
├── AutomaticImageCaptioning.ipynb   model, training, experiments, analysis
├── app.py                           Streamlit app for trying the model
├── models/                          saved checkpoints and training history
└── image/                           images used in this README
```

The Flickr8k dataset is not in the repository. Download it from the link above and place it at `flickr8k/Images/` with `flickr8k/captions.txt`.

## Requirements

- Python 3.10+
- PyTorch and torchvision
- nltk
- streamlit
- numpy, pandas, matplotlib, pillow, tqdm

A CUDA GPU is strongly recommended. The runs above were done on an RTX 4080 SUPER.

## Team Members

- [Reema Al Jbreen](https://github.com/Rmsaah)
- [Madawee AlHathloul](https://github.com/madaweehath)

## Future Work

- Add an attention mechanism so the decoder can look at different parts of the image as it writes each word. This is also where a higher input resolution would start to pay off, since the current encoder averages the whole image into a single vector regardless of how many pixels go in.
- Try a Transformer decoder in place of the LSTM.
- Move to Flickr30k, which is about four times the data.

## Citation and Credits

Ghandi, V., Poovammal, E., & Aarthi, G. (2022). Deep Learning Approaches on Image Captioning. *2022 6th International Conference on Trends in Electronics and Informatics (ICOEI)*, 1076-1082. IEEE. https://doi.org/10.1109/ICOEI53556.2022.9777114

Vinod, S. (2019). A PyTorch Tutorial to Image Captioning (With Attention). GitHub. https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

The AI Epiphany (2022). Image Captioning with Attention - A PyTorch Tutorial Explained. https://www.youtube.com/watch?v=y2BaTt1fxJU
