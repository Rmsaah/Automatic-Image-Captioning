import os  # For checking model file existence
import re
from collections import Counter  # For Vocabulary class

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models  # For EncoderCNN
import torchvision.transforms as transforms
from PIL import Image

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Written by Stage 15 of AutomaticImageCaptioning.ipynb, which copies whichever
# experiment scored best to this stable filename.
MODEL_PATH = "models/best_image_captioning_model.pth"

# Fallbacks, used only when a checkpoint predates these keys being saved.
DEFAULT_EMBED_SIZE = 256
DEFAULT_HIDDEN_SIZE = 512
DEFAULT_NUM_LAYERS_LSTM = 1
DEFAULT_DROPOUT_PROB_DECODER = 0.5

# Beam Search Parameters
DEFAULT_BEAM_WIDTH = 3
DEFAULT_MAX_CAPTION_LENGTH = 50
DEFAULT_LENGTH_PENALTY_ALPHA = 0.7


# --- Model and Vocabulary Definitions (must match the training notebook) ---

class Vocabulary:
    def __init__(self, freq_threshold: int):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {word: idx for idx, word in self.itos.items()}
        self.idx = len(self.itos)

    def __len__(self) -> int:
        return len(self.itos)

    def add_word(self, word: str):
        if word not in self.stoi:
            self.stoi[word] = self.idx
            self.itos[self.idx] = word
            self.idx += 1

    def build_vocabulary(self, sentence_list: list[str]):  # Not used for inference
        frequencies = Counter()
        for sentence in sentence_list:
            for word in re.findall(r'\b\w+\b', sentence.lower()): frequencies[word] += 1
        for word, count in frequencies.items():
            if count >= self.freq_threshold: self.add_word(word)

    def numericalize(self, text: str) -> list[int]:  # Not used for inference
        tokenized_text = re.findall(r'\b\w+\b', text.lower())
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int):
        super(EncoderCNN, self).__init__()
        # weights=None: every parameter is overwritten by the checkpoint a moment later,
        # so downloading the ImageNet weights here would only slow down the first launch.
        resnet = models.resnet50(weights=None)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        return self.bn(self.linear(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int,
                 dropout_prob: float = DEFAULT_DROPOUT_PROB_DECODER):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout_prob if num_layers > 1 else 0.0,
                            batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        embeddings = self.dropout(self.embed(captions))
        full_embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(full_embeddings)
        outputs = self.linear(hiddens)
        return outputs


def generate_caption_beam_search(
        image_tensor: torch.Tensor, encoder: EncoderCNN, decoder: DecoderRNN, vocab: Vocabulary, device: torch.device,
        max_len: int = DEFAULT_MAX_CAPTION_LENGTH, beam_width: int = DEFAULT_BEAM_WIDTH,
        length_penalty_alpha: float = DEFAULT_LENGTH_PENALTY_ALPHA
) -> list[str]:
    encoder.eval();
    decoder.eval()
    with torch.no_grad():
        encoder_output = encoder(image_tensor.unsqueeze(0).to(device))
        initial_features_for_lstm = encoder_output.unsqueeze(0)
        _, current_lstm_states = decoder.lstm(initial_features_for_lstm, None)
        sos_idx, eos_idx, pad_idx = vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]
        sequences = [[[sos_idx], 0.0, current_lstm_states]];
        completed_sequences = []
        for _ in range(max_len):
            all_candidates = [];
            any_beam_not_ended = False
            for seq_indices, score, prev_lstm_states in sequences:
                if seq_indices[-1] == eos_idx:
                    if not any(
                        comp_seq[0] == seq_indices for comp_seq in completed_sequences): completed_sequences.append(
                        (seq_indices, score))
                    continue
                any_beam_not_ended = True
                last_word_idx = seq_indices[-1]
                word_input_tensor = torch.tensor([last_word_idx], device=device)
                embedding_for_lstm = decoder.embed(word_input_tensor).unsqueeze(0)
                hiddens_from_lstm, new_lstm_states = decoder.lstm(embedding_for_lstm, prev_lstm_states)
                output_logits = decoder.linear(hiddens_from_lstm.squeeze(0))
                log_probs = torch.log_softmax(output_logits, dim=1)
                topk_log_probs, topk_indices = log_probs.topk(beam_width, dim=1)
                for i in range(beam_width):
                    next_word_idx = topk_indices[0][i].item()
                    if next_word_idx == pad_idx and beam_width > 1 and len(seq_indices) > 1: continue
                    new_candidate_seq = seq_indices + [topk_indices[0][i].item()]
                    new_candidate_score = score + topk_log_probs[0][i].item()
                    all_candidates.append((new_candidate_seq, new_candidate_score, new_lstm_states))
            if not any_beam_not_ended or not all_candidates: break
            ordered_candidates = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences = ordered_candidates[:beam_width]
        for seq_indices, score, _ in sequences:
            if seq_indices[-1] != eos_idx and not any(comp_seq[0] == seq_indices for comp_seq in completed_sequences):
                completed_sequences.append((seq_indices, score))
        if not completed_sequences:
            best_sequence_indices = sequences[0][0] if sequences and sequences[0][0] else [vocab.stoi["<UNK>"]]
        else:
            def get_penalized_score(seq_data):
                seq, log_prob = seq_data;
                lp_len = max(1, len(seq) - 1)
                lp = ((5 + lp_len) / 6.0) ** length_penalty_alpha
                return log_prob / lp if lp != 0 else float('-inf')

            completed_sequences.sort(key=get_penalized_score, reverse=True)
            best_sequence_indices = completed_sequences[0][0]
        return [vocab.itos[idx] for idx in best_sequence_indices if idx not in {pad_idx, sos_idx, eos_idx}]


# --- Image Transformation ---
app_image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- Load Model and Vocabulary ---
@st.cache_resource
def load_application_data(device: torch.device):
    """Loads the exported checkpoint and rebuilds the models and vocabulary."""
    model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)

    if not os.path.exists(model_path):
        st.error(
            f"Model file not found at {model_path}. Run Stage 15 of "
            f"AutomaticImageCaptioning.ipynb to export a trained model."
        )
        return None, None, None, None

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        st.error(f"Error loading the model checkpoint: {e}")
        return None, None, None, None

    try:
        loaded_vocab = Vocabulary(checkpoint['vocab_freq_threshold'])
        loaded_vocab.stoi = checkpoint['vocab_stoi']
        loaded_vocab.itos = checkpoint['vocab_itos']
        loaded_vocab.idx = len(loaded_vocab.itos)
    except KeyError as e:
        st.error(f"Vocabulary data missing or corrupt in the checkpoint: {e}.")
        return None, None, None, None

    embed_size = checkpoint.get('embed_size', DEFAULT_EMBED_SIZE)
    hidden_size = checkpoint.get('hidden_size', DEFAULT_HIDDEN_SIZE)
    num_layers_lstm = checkpoint.get('num_layers_lstm', DEFAULT_NUM_LAYERS_LSTM)

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(loaded_vocab), num_layers_lstm).to(device)

    try:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    except KeyError as e:
        st.error(f"Model state_dict(s) missing from the checkpoint: {e}.")
        return None, None, None, None
    except RuntimeError as e:
        st.error(f"Error loading weights (architecture mismatch?): {e}")
        return None, None, None, None

    encoder.eval()
    decoder.eval()
    return encoder, decoder, loaded_vocab, checkpoint


# --- Streamlit App UI ---
st.set_page_config(page_title="Image Captioner", layout="wide")
st.title("Image Caption Generator")
st.markdown("Upload an image and the model will describe it.")

current_encoder, current_decoder, current_vocabulary, model_info = load_application_data(DEVICE)

if current_encoder is None:
    st.stop()

experiment = model_info.get('config', {}).get('name', 'unknown')
st.caption(
    f"Model: {experiment} (epoch {model_info.get('epoch', '?')}) · "
    f"BLEU-4 {model_info.get('best_val_bleu4', 0):.4f} · "
    f"beam width {DEFAULT_BEAM_WIDTH} · running on {DEVICE}"
)
st.markdown("---")

uploaded_file = st.file_uploader("Choose an image file (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")

        st.subheader("Your Uploaded Image")
        st.image(image_pil, caption="Uploaded Image", width=image_pil.size[0])
        st.markdown("---")

        if st.button("Generate Caption"):
            with st.spinner("Generating..."):
                img_tensor_transformed = app_image_transform(image_pil)

                caption_tokens_list = generate_caption_beam_search(
                    img_tensor_transformed, current_encoder, current_decoder, current_vocabulary, DEVICE
                )
                generated_caption_text = " ".join(caption_tokens_list)

            st.subheader("Generated Caption")
            if generated_caption_text:
                st.markdown(f"## _{generated_caption_text.capitalize()}_")
            else:
                st.warning("The model could not generate a caption for this image. Try another one.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Upload an image to get started.")

st.markdown("---")
st.markdown("Built with PyTorch and Streamlit.")
