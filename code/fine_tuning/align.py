import os
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from peft import get_peft_model, LoraConfig, TaskType

# Configuration parameters
MODEL_NAME = "openai-community/gpt2"
DATA_PATH = "../../data/hedging_questions.csv"
SAVE_DIR = f"fine_tuned_{MODEL_NAME.replace('/', '_')}"
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_LENGTH = 512
RATING_ALIGNMENT_WEIGHT = 2.0      # Weight for rating alignment loss
DECISION_ALIGNMENT_WEIGHT = 1.0      # Weight for final decision alignment loss
REPRESENTATION_ALIGNMENT_WEIGHT = 1.0  # Weight for representation (hidden state) alignment loss
USE_LORA = True                    # Set to True to use LoRA

# Device selection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# Evaluation prompt components for rating alignment
EVALUATION_CATEGORIES = [
    "Educational Background: Does the candidate have the appropriate educational qualifications or training for this position?",
    "Job/Organizational Fit: Has the candidate acquired similar skills or qualifications through past work experiences?",
    "Problem Solving: Did the candidate display the ability to react to a situation and devise a strategy?",
    "Verbal Communication: How were the candidate's communication skills during the interview?",
    "Candidate Interest: How much interest did the candidate show in the position and the organization?",
    "Knowledge of Organization: Did the candidate research the organization prior to the interview?",
    "Teambuilding/Interpersonal Skills: Did the candidate demonstrate, through their answers, good teambuilding/interpersonal skills?",
    "Initiative: - Did the candidate demonstrate, through their answers, a high degree of initiative?",
    "Time Management: Did the candidate demonstrate, through their answers, good time management skills?",
    "Attention to Detail: Was the candidate able to give provide examples of detail in their previous work experience?"
]

EVAL_TEMPLATE = """
You are a hiring manager evaluating a candidate's response to an interview question. You want to hire the best candidate for the job and are going to think critically and rate the candidate's response.

Evaluate the interview response with the following categories in mind:
{}

You will rate the interview response on a scale from 1 to 5 based on the content of the response, using the following scale:
5 - Far Exceeds Requirements: Perfect answer. Demonstrates competency accurately, consistently, and independently. All points relevant. The answer includes only good and relavant examples.
4 - Exceeds Requirements: Demonstrates competency accurately and consistently in most situations with minimal guidance. The answer would benift from examples and many good examples provided.
3 - Meets Requirements: Demonstrates competency accurately and consistently on familiar procedures and needs supervisor guidance for new skills. The answer would benift from examples and some good examples provided.
2 - Below Requirements: Demonstrates competency inconsistently, even with repeated instruction or guidance. Although the answer would benift from examples, few good examples provided.
1 - Significant Gap: Fails to demonstrate competency regardless of guidance provided. Although the answer would benift from examples, no good examples provided.

It is extremely important that as a response you provide a clear, detailed, and conscise explanation along with the rating.
Provide your response in the following format:
Reasoning: <your reasoning>
Rating: <your rating>

Context:
The question that was asked was: "{question}"
The interview response was: "{response}"
Rating: """

FINAL_DECISION_TEMPLATE = """
You are a hiring manager evaluating a candidate's responses to an interview. You have asked the candidate a series of questions and rated their responses for each question from 1 to 5.

Based on the following interview questions and the candidate's responses, determine whether the candidate should:
- advance
- advance with reservations
- do not advance

For your response, provide your explanation and then simply output one of the following tokens as your final decision: "advance", "reservations", or "reject". These correspond to "advance", "advance with reservations", and "do not advance", respectively.

Provide your response in the following format:
Reasoning: <your reasoning>
Result: <your result>

Context:
The questions and the candidate's responses are:
{responses}
Result: """

# Precompute token ids for rating digits "1" to "5".
def get_rating_token_ids(tokenizer):
    rating_digits = ["1", "2", "3", "4", "5"]
    return torch.tensor([tokenizer.convert_tokens_to_ids(d) for d in rating_digits], device=DEVICE)

# Precompute token ids for decision tokens: "advance", "reservations", "reject"
def get_decision_token_ids(tokenizer):
    decision_tokens = ["advance", "reservations", "reject"]
    return torch.tensor([tokenizer.convert_tokens_to_ids(tok) for tok in decision_tokens], device=DEVICE)

# Custom dataset that returns raw text.
class QADataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "question": sample["question"],
            "hedged_text": sample["answer_hedged"],
            "confident_text": sample["answer_confident"]
        }

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

def collate_fn(batch):
    return batch

# --- Existing Loss Functions ---

def compute_rating_alignment_loss(prompt_texts_1, prompt_texts_2, model, tokenizer, rating_token_ids):
    enc1 = tokenizer(prompt_texts_1, return_tensors="pt", padding=True)
    enc2 = tokenizer(prompt_texts_2, return_tensors="pt", padding=True)
    for key in enc1: enc1[key] = enc1[key].to(DEVICE)
    for key in enc2: enc2[key] = enc2[key].to(DEVICE)
    
    with torch.no_grad():
        lengths1 = enc1["attention_mask"].sum(dim=1)
        lengths2 = enc2["attention_mask"].sum(dim=1)
    
    outputs1 = model(**enc1)
    outputs2 = model(**enc2)
    logits1 = outputs1.logits
    logits2 = outputs2.logits

    batch_size = logits1.size(0)
    rating_logits_1 = []
    rating_logits_2 = []
    for i in range(batch_size):
        idx1 = lengths1[i].item() - 1
        idx2 = lengths2[i].item() - 1
        token_logits1 = logits1[i, idx1, :]
        token_logits2 = logits2[i, idx2, :]
        rating_logits_1.append(token_logits1[rating_token_ids])
        rating_logits_2.append(token_logits2[rating_token_ids])
    rating_logits_1 = torch.stack(rating_logits_1, dim=0)
    rating_logits_2 = torch.stack(rating_logits_2, dim=0)

    p1 = F.softmax(rating_logits_1, dim=-1)
    p2 = F.softmax(rating_logits_2, dim=-1)
    kl1 = F.kl_div(p1.log(), p2, reduction="batchmean")
    kl2 = F.kl_div(p2.log(), p1, reduction="batchmean")
    loss = kl1 + kl2
    return loss

def compute_decision_alignment_loss(prompt_texts_1, prompt_texts_2, model, tokenizer, decision_token_ids):
    enc1 = tokenizer(prompt_texts_1, return_tensors="pt", padding=True)
    enc2 = tokenizer(prompt_texts_2, return_tensors="pt", padding=True)
    for key in enc1: enc1[key] = enc1[key].to(DEVICE)
    for key in enc2: enc2[key] = enc2[key].to(DEVICE)
    
    with torch.no_grad():
        lengths1 = enc1["attention_mask"].sum(dim=1)
        lengths2 = enc2["attention_mask"].sum(dim=1)
    
    outputs1 = model(**enc1)
    outputs2 = model(**enc2)
    logits1 = outputs1.logits
    logits2 = outputs2.logits

    batch_size = logits1.size(0)
    decision_logits_1 = []
    decision_logits_2 = []
    for i in range(batch_size):
        idx1 = lengths1[i].item() - 1
        idx2 = lengths2[i].item() - 1
        token_logits1 = logits1[i, idx1, :]
        token_logits2 = logits2[i, idx2, :]
        decision_logits_1.append(token_logits1[decision_token_ids])
        decision_logits_2.append(token_logits2[decision_token_ids])
    decision_logits_1 = torch.stack(decision_logits_1, dim=0)
    decision_logits_2 = torch.stack(decision_logits_2, dim=0)

    p1 = F.softmax(decision_logits_1, dim=-1)
    p2 = F.softmax(decision_logits_2, dim=-1)
    kl1 = F.kl_div(p1.log(), p2, reduction="batchmean")
    kl2 = F.kl_div(p2.log(), p1, reduction="batchmean")
    loss = kl1 + kl2
    return loss

# --- New Loss Function: Representation Alignment Loss ---
def compute_hidden_state_alignment_loss(prompt_texts_1, prompt_texts_2, model, tokenizer):
    enc1 = tokenizer(prompt_texts_1, return_tensors="pt", padding=True)
    enc2 = tokenizer(prompt_texts_2, return_tensors="pt", padding=True)
    for key in enc1: enc1[key] = enc1[key].to(DEVICE)
    for key in enc2: enc2[key] = enc2[key].to(DEVICE)
    
    outputs1 = model(**enc1, output_hidden_states=True)
    outputs2 = model(**enc2, output_hidden_states=True)
    lengths1 = enc1["attention_mask"].sum(dim=1)
    lengths2 = enc2["attention_mask"].sum(dim=1)
    
    hs1 = []
    hs2 = []
    for i in range(enc1["input_ids"].size(0)):
        hs1.append(outputs1.hidden_states[-1][i, lengths1[i]-1, :])
        hs2.append(outputs2.hidden_states[-1][i, lengths2[i]-1, :])
    hs1 = torch.stack(hs1, dim=0)
    hs2 = torch.stack(hs2, dim=0)
    
    cosine_loss = 1 - F.cosine_similarity(hs1, hs2, dim=-1).mean()
    mse_loss = F.mse_loss(hs1, hs2)
    return cosine_loss + mse_loss

def train_alignment():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    if USE_LORA:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)

    data = load_data(DATA_PATH)
    dataset = QADataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    rating_token_ids = get_rating_token_ids(tokenizer)
    decision_token_ids = get_decision_token_ids(tokenizer)

    # --- Warm-up Step ---
    start_time = time.time()
    model.eval()
    dummy_prompt_rating = EVAL_TEMPLATE.format("\n".join(EVALUATION_CATEGORIES), question="dummy", response="dummy")
    dummy_prompt_decision = FINAL_DECISION_TEMPLATE.format(responses="Question: dummy\nResponse: dummy")
    with torch.no_grad():
        _ = compute_rating_alignment_loss([dummy_prompt_rating], [dummy_prompt_rating], model, tokenizer, rating_token_ids)
        _ = compute_decision_alignment_loss([dummy_prompt_decision], [dummy_prompt_decision], model, tokenizer, decision_token_ids)
        _ = compute_hidden_state_alignment_loss([dummy_prompt_rating], [dummy_prompt_rating], model, tokenizer)
    model.train()
    warmup_duration = time.time() - start_time
    print(f"Warm-up complete in {warmup_duration:.2f} sec")
    # --- End Warm-up ---

    epoch_losses = []
    epoch_times = []
    batch_loss_history = []
    global_batch_count = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            optimizer.zero_grad()
            # Build prompts.
            eval_prompts_hedged = []
            eval_prompts_confident = []
            final_decision_prompts_hedged = []
            final_decision_prompts_confident = []
            for sample in batch:
                prompt_hedged = EVAL_TEMPLATE.format(
                    "\n".join(EVALUATION_CATEGORIES),
                    question=sample["question"],
                    response=sample["hedged_text"]
                )
                prompt_confident = EVAL_TEMPLATE.format(
                    "\n".join(EVALUATION_CATEGORIES),
                    question=sample["question"],
                    response=sample["confident_text"]
                )
                eval_prompts_hedged.append(prompt_hedged)
                eval_prompts_confident.append(prompt_confident)
                responses_context_hedged = f"Question: {sample['question']}\nHedged Response: {sample['hedged_text']}"
                responses_context_confident = f"Question: {sample['question']}\nConfident Response: {sample['confident_text']}"
                final_prompt_hedged = FINAL_DECISION_TEMPLATE.format(responses=responses_context_hedged)
                final_prompt_confident = FINAL_DECISION_TEMPLATE.format(responses=responses_context_confident)
                final_decision_prompts_hedged.append(final_prompt_hedged)
                final_decision_prompts_confident.append(final_prompt_confident)
            
            rating_loss = RATING_ALIGNMENT_WEIGHT * compute_rating_alignment_loss(
                eval_prompts_hedged, eval_prompts_confident, model, tokenizer, rating_token_ids
            )
            decision_loss = DECISION_ALIGNMENT_WEIGHT * compute_decision_alignment_loss(
                final_decision_prompts_hedged, final_decision_prompts_confident, model, tokenizer, decision_token_ids
            )
            rep_loss = REPRESENTATION_ALIGNMENT_WEIGHT * compute_hidden_state_alignment_loss(
                eval_prompts_hedged, eval_prompts_confident, model, tokenizer
            )
            
            loss = rating_loss + decision_loss + rep_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_loss_history.append(batch_loss)
            global_batch_count += 1
        epoch_duration = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_duration)
        print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f} | Time: {epoch_duration:.2f} sec")

    # Plot epoch-level average losses.
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss')
    plt.title('Epoch-Level Loss Progression')

    # Plot batch-level loss history.
    plt.subplot(1, 2, 2)
    plt.plot(batch_loss_history, label='Batch Loss')
    plt.xlabel('Batch Iteration')
    plt.ylabel('Loss')
    plt.title('Batch-Level Loss Progression')
    plt.legend()
    plt.tight_layout()
    plt.show()

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved to {SAVE_DIR}")

if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
    train_alignment()
