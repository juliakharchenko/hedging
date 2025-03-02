import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration parameters
MODEL_NAME = "openai-community/gpt2"  # Change this as needed
DATA_PATH = "../../data/hedging_questions.csv"
SAVE_DIR = f"fine_tuned_{MODEL_NAME.replace('/', '_')}"
BATCH_SIZE = 2                # Lower batch size helps with stability and memory
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
MAX_LENGTH = 512
LAMBDA = 0.75                 # Weight for LM loss
ALPHA = 9.0                   # Weight for cosine loss on hidden states
BETA = 11.0                   # Weight for KL divergence loss on logits
GAMMA_COS_LOGITS = 1.0        # Weight for cosine loss on logits
GAMMA_KL_HIDDEN = 1.0         # Weight for KL divergence loss on hidden states

# Device selection with MPS support for Mac (or CUDA/CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

def load_data(file_path):
    df = pd.read_csv(file_path)
    data = []
    for _, row in df.iterrows():
        data.append({
            "question": row["Question"],
            "answer_hedged": row["Hedging_Answer"],
            "answer_confident": row["Confident_Answer"]
        })
    return data

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample["question"]
        answer_hedged = sample["answer_hedged"]
        answer_confident = sample["answer_confident"]

        # Tokenize hedged and confident responses
        enc_hedged = self.tokenizer(
            question + " " + answer_hedged,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        enc_confident = self.tokenizer(
            question + " " + answer_confident,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # Remove batch dimension
        for key in enc_hedged:
            enc_hedged[key] = enc_hedged[key].squeeze(0)
        for key in enc_confident:
            enc_confident[key] = enc_confident[key].squeeze(0)

        # Create labels by cloning input_ids and masking pad tokens with -100
        enc_hedged["labels"] = enc_hedged["input_ids"].clone()
        enc_confident["labels"] = enc_confident["input_ids"].clone()
        enc_hedged["labels"][enc_hedged["labels"] == self.tokenizer.pad_token_id] = -100
        enc_confident["labels"][enc_confident["labels"] == self.tokenizer.pad_token_id] = -100

        return {"hedged": enc_hedged, "confident": enc_confident}

def collate_fn(batch):
    batch_hedged, batch_confident = {}, {}
    for key in batch[0]["hedged"]:
        batch_hedged[key] = torch.stack([sample["hedged"][key] for sample in batch])
        batch_confident[key] = torch.stack([sample["confident"][key] for sample in batch])
    return {"hedged": batch_hedged, "confident": batch_confident}

def evaluate_alignment(model, dataset, device, num_samples=10):
    """
    Evaluate the average cosine similarity and average KL divergence between the output distributions 
    of the hedged and confident responses.
    Here we compute these metrics on both the last hidden states and the logits.
    """
    model.eval()
    cos_sims_hidden = []
    cos_sims_logits = []
    kl_divs_hidden = []
    kl_divs_logits = []
    eps = 1e-8
    with torch.no_grad():
        for idx in range(min(num_samples, len(dataset))):
            sample = dataset[idx]
            hedged_input = {k: v.unsqueeze(0).to(device) for k, v in sample["hedged"].items()}
            confident_input = {k: v.unsqueeze(0).to(device) for k, v in sample["confident"].items()}

            outputs_hedged = model(**hedged_input, output_hidden_states=True)
            outputs_confident = model(**confident_input, output_hidden_states=True)

            # Hidden states from last token
            hidden_hedged = outputs_hedged.hidden_states[-1][:, -1, :].to(torch.float32)
            hidden_confident = outputs_confident.hidden_states[-1][:, -1, :].to(torch.float32)
            cos_hidden = F.cosine_similarity(hidden_hedged, hidden_confident, dim=-1).item()
            cos_sims_hidden.append(cos_hidden)
            # KL divergence on hidden states: first normalize with softmax and clamp
            p_hidden_hedged = torch.clamp(torch.softmax(hidden_hedged, dim=-1), min=eps)
            p_hidden_confident = torch.clamp(torch.softmax(hidden_confident, dim=-1), min=eps)
            kl_hidden = (F.kl_div(p_hidden_hedged.log(), p_hidden_confident, reduction='batchmean') +
                         F.kl_div(p_hidden_confident.log(), p_hidden_hedged, reduction='batchmean')) / 2
            kl_divs_hidden.append(kl_hidden.item())

            # Logits from last token
            logits_hedged = outputs_hedged.logits[:, -1, :]
            logits_confident = outputs_confident.logits[:, -1, :]
            cos_logits = F.cosine_similarity(logits_hedged, logits_confident, dim=-1).item()
            cos_sims_logits.append(cos_logits)
            # KL divergence on logits: compute softmax and clamp
            p_hedged = torch.clamp(torch.softmax(logits_hedged, dim=-1), min=eps)
            p_confident = torch.clamp(torch.softmax(logits_confident, dim=-1), min=eps)
            kl_logits = (F.kl_div(p_hedged.log(), p_confident, reduction='batchmean') +
                         F.kl_div(p_confident.log(), p_hedged, reduction='batchmean')) / 2
            kl_divs_logits.append(kl_logits.item())

    model.train()
    avg_cos_sim_hidden = sum(cos_sims_hidden) / len(cos_sims_hidden) if cos_sims_hidden else 0.0
    avg_cos_sim_logits = sum(cos_sims_logits) / len(cos_sims_logits) if cos_sims_logits else 0.0
    avg_kl_div_hidden = sum(kl_divs_hidden) / len(kl_divs_hidden) if kl_divs_hidden else 0.0
    avg_kl_div_logits = sum(kl_divs_logits) / len(kl_divs_logits) if kl_divs_logits else 0.0
    return avg_cos_sim_hidden, avg_cos_sim_logits, avg_kl_div_hidden, avg_kl_div_logits

def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Ensure a pad token is set (some models may not have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    # Load data and create dataloader
    data = load_data(DATA_PATH)
    dataset = QADataset(data, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Lists for logging metrics
    epoch_total_losses = []
    epoch_lm_losses = []
    epoch_cos_losses_hidden = []
    epoch_cos_losses_logits = []
    epoch_kl_losses_logits = []
    epoch_kl_losses_hidden = []
    epoch_alignments_hidden = []      # (avg cosine similarity on hidden states)
    epoch_alignments_logits = []      # (avg cosine similarity on logits)

    eps = 1e-8
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_lm_loss = 0.0
        epoch_cos_loss_hidden = 0.0
        epoch_cos_loss_logits = 0.0
        epoch_kl_loss_logits = 0.0
        epoch_kl_loss_hidden = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            optimizer.zero_grad()
            # Move inputs to device
            for key in batch["hedged"]:
                batch["hedged"][key] = batch["hedged"][key].to(DEVICE)
            for key in batch["confident"]:
                batch["confident"][key] = batch["confident"][key].to(DEVICE)

            outputs_hedged = model(**batch["hedged"], output_hidden_states=True)
            outputs_confident = model(**batch["confident"], output_hidden_states=True)

            # LM losses
            loss_hedged = outputs_hedged.loss
            loss_confident = outputs_confident.loss
            lm_loss = LAMBDA * ((loss_hedged + loss_confident) / 2)

            # Cosine similarity loss on hidden states
            hidden_hedged = outputs_hedged.hidden_states[-1][:, -1, :].to(torch.float32)
            hidden_confident = outputs_confident.hidden_states[-1][:, -1, :].to(torch.float32)
            cos_loss_hidden = ALPHA * (1 - F.cosine_similarity(hidden_hedged, hidden_confident, dim=-1).mean())

            # Cosine similarity loss on logits
            logits_hedged = outputs_hedged.logits[:, -1, :]
            logits_confident = outputs_confident.logits[:, -1, :]
            cos_loss_logits = GAMMA_COS_LOGITS * (1 - F.cosine_similarity(logits_hedged, logits_confident, dim=-1).mean())

            # KL divergence loss on logits: softmax, clamp, then symmetric KL
            p_hedged = torch.clamp(torch.softmax(logits_hedged, dim=-1), min=eps)
            p_confident = torch.clamp(torch.softmax(logits_confident, dim=-1), min=eps)
            kl_loss_logits = BETA * ((F.kl_div(p_hedged.log(), p_confident, reduction='batchmean') +
                                      F.kl_div(p_confident.log(), p_hedged, reduction='batchmean')) / 2)

            # KL divergence loss on hidden states: softmax, clamp, then symmetric KL
            p_hidden_hedged = torch.clamp(torch.softmax(hidden_hedged, dim=-1), min=eps)
            p_hidden_confident = torch.clamp(torch.softmax(hidden_confident, dim=-1), min=eps)
            kl_loss_hidden = GAMMA_KL_HIDDEN * ((F.kl_div(p_hidden_hedged.log(), p_hidden_confident, reduction='batchmean') +
                                                F.kl_div(p_hidden_confident.log(), p_hidden_hedged, reduction='batchmean')) / 2)

            total_loss = lm_loss + cos_loss_hidden + cos_loss_logits + kl_loss_logits + kl_loss_hidden
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_lm_loss += lm_loss.item()
            epoch_cos_loss_hidden += cos_loss_hidden.item()
            epoch_cos_loss_logits += cos_loss_logits.item()
            epoch_kl_loss_logits += kl_loss_logits.item()
            epoch_kl_loss_hidden += kl_loss_hidden.item()

        avg_total_loss = epoch_loss / len(dataloader)
        avg_lm_loss = epoch_lm_loss / len(dataloader)
        avg_cos_loss_hidden = epoch_cos_loss_hidden / len(dataloader)
        avg_cos_loss_logits = epoch_cos_loss_logits / len(dataloader)
        avg_kl_loss_logits = epoch_kl_loss_logits / len(dataloader)
        avg_kl_loss_hidden = epoch_kl_loss_hidden / len(dataloader)

        # Evaluate alignment on a subset of samples
        avg_cos_sim_hidden, avg_cos_sim_logits, avg_kl_div_hidden, avg_kl_div_logits = evaluate_alignment(model, dataset, DEVICE, num_samples=10)
        print(f"Epoch {epoch + 1} | Total Loss: {avg_total_loss:.4f} | LM Loss: {avg_lm_loss:.4f} | "
              f"Cosine Loss Hidden: {avg_cos_loss_hidden:.4f} | Cosine Loss Logits: {avg_cos_loss_logits:.4f} | "
              f"KL Loss Logits: {avg_kl_loss_logits:.4f} | KL Loss Hidden: {avg_kl_loss_hidden:.4f} | "
              f"Avg Alignment (cos-hidden): {avg_cos_sim_hidden:.4f} | Avg Alignment (cos-logits): {avg_cos_sim_logits:.4f} | "
              f"Avg KL Div (hidden): {avg_kl_div_hidden:.4f} | Avg KL Div (logits): {avg_kl_div_logits:.4f}")

        epoch_total_losses.append(avg_total_loss)
        epoch_lm_losses.append(avg_lm_loss)
        epoch_cos_losses_hidden.append(avg_cos_loss_hidden)
        epoch_cos_losses_logits.append(avg_cos_loss_logits)
        epoch_kl_losses_logits.append(avg_kl_loss_logits)
        epoch_kl_losses_hidden.append(avg_kl_loss_hidden)

    # Save the fine-tuned model and tokenizer
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved to {SAVE_DIR}")

    # Plot and save training metrics
    epochs = list(range(1, NUM_EPOCHS + 1))
    plt.figure(figsize=(12, 6))
    # Subplot 1: Loss metrics over epochs
    plt.subplot(1, 2, 1)
    plt.plot(epochs, epoch_total_losses, label="Total Loss")
    plt.plot(epochs, epoch_lm_losses, label="LM Loss")
    plt.plot(epochs, epoch_cos_losses_hidden, label="Cosine Loss Hidden")
    plt.plot(epochs, epoch_cos_losses_logits, label="Cosine Loss Logits")
    plt.plot(epochs, epoch_kl_losses_logits, label="KL Loss Logits")
    plt.plot(epochs, epoch_kl_losses_hidden, label="KL Loss Hidden")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Metrics")
    plt.legend()

    # Subplot 2: Alignment metrics over epochs
    plt.subplot(1, 2, 2)
    # Here we show the average cosine similarities computed in evaluation
    # (You could also plot the KL divergence values if desired)
    plt.plot(epochs, [evaluate_alignment(model, dataset, DEVICE, num_samples=10)[0] for _ in epochs], label="Avg Cosine (hidden)")
    plt.plot(epochs, [evaluate_alignment(model, dataset, DEVICE, num_samples=10)[1] for _ in epochs], label="Avg Cosine (logits)")
    plt.xlabel("Epoch")
    plt.ylabel("Alignment Metric")
    plt.title("Alignment Metrics")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_metrics.png"))
    plt.show()

if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
    train()
