# 🖼️ Automatic Image Captioning

This repository contains a deep learning project that generates descriptive captions for images. It uses a Convolutional Neural Network (CNN) as an encoder to extract visual features and a Recurrent Neural Network (RNN) with LSTM architecture as a decoder to produce natural language captions.

---

## Project Structure

```
.
├── AutomaticImageCaptioning.ipynb  # Jupyter notebook: model implementation, training, and evaluation
├── app.py                          # Streamlit web app for testing models with different BLEU optimizations
```

---

##  Model Architecture

### Encoder

* **Model**: Pre-trained ResNet50 (`ResNet50_Weights.IMAGENET1K_V1`)
* **Modifications**: Final classification layer replaced with a linear layer to project into embedding space
* **Training**: CNN layers are initially frozen, with optional fine-tuning
* **Output**: Visual feature embedding

### Decoder

* **Model**: LSTM
* **Embedding Size**: 256
* **Hidden Size**: 512
* **Layers**: 1
* **Dropout**: 0.5 before the final linear layer

### Vocabulary

* Built using a frequency threshold of 5
* Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`

---

##  Dataset

* **Source**: [Kaggle - Flickr 8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
* **Content**: 8,000 images with 5 captions each
* **Split**: 80% training, 20% validation

---

##  Image Preprocessing

| Step                | Description                                                                     |
| ------------------- | ------------------------------------------------------------------------------- |
| **Resize**          | All images resized to 256x256                                                   |
| **Random Crop**     | `RandomCrop(224, 224)` applied during training                                  |
| **Flip**            | Random horizontal flips for augmentation                                        |
| **Normalization**   | Mean = `[0.485, 0.456, 0.406]`, Std = `[0.229, 0.224, 0.225]` (ImageNet values) |
| **Validation/Test** | Resize to 224x224 + normalization only                                          |
| **Unnormalization** | Utility provided for visualization                                              |

---

##  Training Details

* **Epochs**: 10
* **Batch Size**: 32
* **Learning Rate**: `3e-4`
* **Optimizer**: Adam
* **Loss Function**: `nn.CrossEntropyLoss` (ignores `<PAD>`)
* **Scheduler**: `StepLR` (reduce LR every 5 epochs by 0.1)
* **Checkpoint**: Best model (`best_image_captioning_model.pth`) saved based on BLEU-4
* **Reproducibility**: Random seeds set for `random`, `numpy`, and `torch`

---

## Evaluation Metric

The model's performance is rigorously evaluated using the BLEU (Bilingual Evaluation Understudy) score. BLEU measures the similarity between the generated caption and a set of reference captions, focusing on n-gram precision. During our evaluation, we assessed the model's performance using BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores to capture different levels of n-gram precision. A smoothing function (SmoothingFunction().method4) is applied during BLEU calculation to handle cases with no n-gram matches, ensuring scores are well-defined even for short sequences.
* **Metric**: BLEU Score (Bilingual Evaluation Understudy)
* **Variants**: BLEU-1 to BLEU-4
* **Smoothing**: `SmoothingFunction().method4` applied for short sequences

---

## Results

* **Best Validation BLEU-4 Score**: `0.0199`
* Captions were generated and evaluated on the validation set.

| Image         | BLEU-1                       | BLEU-2                 | BLEU-3               | BLEU-4            |
| ------------- | ---------------------------- | ---------------------- | -------------------- | ----------------- |
| !(image/img1.png)  | Dogs running the             | Dog a dog a ball a     | Dogs in field a      | Woman a and dog a |
| !(image/img2.png) |Man a and woman a and woman a | Man a <unk> a          | Man a and woman a    | Man a in and shirt a |
| !(image/img3.png)  | Man a on bike a              | Boy in red is on beach | Man a on bike a      | Man on bike a     |



> **Note**: You can run the app to generate these outputs

---

## Web Application

Run `app.py` to launch the **Streamlit web app**, which allows users to upload images and view generated captions from different model versions (BLEU-1 to BLEU-4 optimized).

```bash
streamlit run app.py
```

---

## Literature Review & Future Work

This project is inspired by the paper:
Our image captioning system is built upon the widely recognized Encoder-Decoder architecture, drawing inspiration from the review paper "Deep Learning Approaches on Image Captioning" by Ghandi et al. This paper effectively describes image captioning as a sequence-to-sequence (seq2seq) problem, wherein a CNN encodes visual information and an RNN decodes it into a natural language sequence.

A critical advancement highlighted in the literature is the integration of attention mechanisms into the CNN-RNN pipeline. Attention allows the model to dynamically focus on different parts of the image as it generates each word, leading to more accurate, contextually relevant, and descriptive captions.As part of our research, we reviewed an excellent PyTorch implementation of this attention-based approach. Due to constraints in computational resources and training time, we were unable to implement or train this model. 

---

## 📌 Requirements

* Python 3.8+
* PyTorch
* torchvision
* nltk
* Streamlit
* numpy, matplotlib, PIL, etc.

---

## ✅ To Do

* [ ] Add sample generated image-caption pairs to README
* [ ] Improve BLEU-4 score via attention mechanisms
* [ ] Experiment with alternative decoders (e.g., Transformers)

---

## 💬 Citation & Credits
Ghandi, V., Poovammal, E., & Aarthi, G. (2022). Deep Learning Approaches on Image Captioning. 2022 6th International Conference on Trends in Electronics and Informatics (ICOEI), 1076–1082. IEEE. https://doi.org/10.1109/ICOEI53556.2022.9777114
Vinod, S. (2019). A PyTorch Tutorial to Image Captioning (With Attention). GitHub Repository. https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
The AI Epiphany. (2022, May 13). Image Captioning with Attention - A PyTorch Tutorial Explained [Video]. YouTube. https://www.youtube.com/watch?v=y2BaTt1fxJU
