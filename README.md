# Automatic-Image-Captioning
🖼️ Automatic Image Captioning
This repository contains a deep learning project that generates descriptive captions for images. It uses a Convolutional Neural Network (CNN) as an encoder to extract visual features and a Recurrent Neural Network (RNN) with LSTM architecture as a decoder to produce natural language captions.

📁 Project Structure
bash
Copy
Edit
.
├── AutomaticImageCaptioning.ipynb  # Jupyter notebook: model implementation, training, and evaluation
├── app.py                          # Streamlit web app for testing models with different BLEU optimizations
🧠 Model Architecture
Encoder
Model: Pre-trained ResNet50 (ResNet50_Weights.IMAGENET1K_V1)

Modifications: Final classification layer replaced with a linear layer to project into embedding space

Training: CNN layers are initially frozen, with optional fine-tuning

Output: Visual feature embedding

Decoder
Model: LSTM

Embedding Size: 256

Hidden Size: 512

Layers: 1

Dropout: 0.5 before the final linear layer

Vocabulary
Built using a frequency threshold of 5

Special tokens: <PAD>, <SOS>, <EOS>, <UNK>

🖼️ Dataset
Source: Kaggle - Flickr 8k Dataset

Content: 8,000 images with 5 captions each

Split: 80% training, 20% validation

🧪 Image Preprocessing
Step	Description
Resize	All images resized to 256x256
Random Crop	RandomCrop(224, 224) applied during training
Flip	Random horizontal flips for augmentation
Normalization	Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225] (ImageNet values)
Validation/Test	Resize to 224x224 + normalization only
Unnormalization	Utility provided for visualization

⚙️ Training Details
Epochs: 10

Batch Size: 32

Learning Rate: 3e-4

Optimizer: Adam

Loss Function: nn.CrossEntropyLoss (ignores <PAD>)

Scheduler: StepLR (reduce LR every 5 epochs by 0.1)

Checkpoint: Best model (best_image_captioning_model.pth) saved based on BLEU-4

Reproducibility: Random seeds set for random, numpy, and torch

📏 Evaluation Metric
Metric: BLEU Score (Bilingual Evaluation Understudy)

Variants: BLEU-1 to BLEU-4

Smoothing: SmoothingFunction().method4 applied for short sequences

📊 Results
Best Validation BLEU-4 Score: 0.0199

Captions were generated and evaluated on the validation set.

Image	BLEU-1	BLEU-2	BLEU-3	BLEU-4
Dogs running	Dog a dog a ball a	Dogs in field a	Woman a and dog a	Woman a and dog a
Man and woman	Man a <unk> a	Man a and woman a	Man a in and shirt a	-
Man on bike	Boy in red is on beach	Man on bike a	-	-

Note: You can run the notebook to generate these outputs using plt.show() and save the images + predictions for a visual showcase.

🌐 Web Application
Run app.py to launch the Streamlit web app, which allows users to upload images and view generated captions from different model versions (BLEU-1 to BLEU-4 optimized).

bash
Copy
Edit
streamlit run app.py
📚 Literature Review
This project is inspired by the paper:
"Deep Learning Approaches on Image Captioning" by Ghandi et al.
It views image captioning as a sequence-to-sequence (seq2seq) problem using CNNs to encode image features and RNNs to decode them into text.

🔭 Future Work
Attention Mechanisms: Integrating attention into the Encoder-Decoder pipeline to allow the model to dynamically focus on image regions during generation.

Reference: PyTorch attention-based captioning implementations were reviewed, though not implemented due to resource constraints.

📌 Requirements
Python 3.8+

PyTorch

torchvision

nltk

Streamlit

numpy, matplotlib, PIL, etc.

✅ To Do
 Add sample generated image-caption pairs to README

 Improve BLEU-4 score via attention mechanisms

 Experiment with alternative decoders (e.g., Transformers)

