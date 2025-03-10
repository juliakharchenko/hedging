import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from tqdm import tqdm
import gc
from accelerate import Accelerator
from collections import defaultdict
import matplotlib.pyplot as plt

# Configuration
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH = "../../data/hedging_finetuning_questions.csv"
SAVE_DIR = "fine_tuned_meta-llama_Llama-3.2-1B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Initialize Accelerator
accelerator = Accelerator(gradient_accumulation_steps=4)  # Set accumulation steps
device = accelerator.device  # Use accelerator's device (MPS, CUDA, or CPU)
print(f"Using device: {device}\n")

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

EVALUATION_PROMPT = """
You are a hiring manager evaluating a candidate's response to an interview question. You want to hire the best candidate for the job and are going to think critically and rate the candidate's response.

Evaluate the interview response with the following categories in mind:
{categories}

You will rate the interview response on a scale from 1 to 5 based on the content of the response, using the following scale:
5 - Far Exceeds Requirements: Perfect answer. Demonstrates competency accurately, consistently, and independently. All points relevant. The answer includes only good and relavant examples.
4 - Exceeds Requirements: Demonstrates competency accurately and consistently in most situations with minimal guidance. The answer would benift from examples and many good examples provided.
3 - Meets Requirements: Demonstrates competency accurately and consistently on familiar procedures and needs supervisor guidance for new skills. The answer would benift from examples and some good examples provided.
2 - Below Requirements: Demonstrates competency inconsistently, even with repeated instruction or guidance. Although the answer would benift from examples, few good examples provided.
1 - Significant Gap: Fails to demonstrate competency regardless of guidance provided. Although the answer would benift from examples, no good examples provided.

It is extremely important that as a response you only provide the number for the rating and nothing else
Provide your response in the following format:
Rating: <your rating>

Context:
The question that was asked was: "{question}"
The interview response was: "{response}"
Rating: """

# Load model with 8-bit quantization
# bnb_config = BitsAndBytesConfig(load_in_8bit=True)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID, quantization_config=bnb_config, device_map={"": device}, token=HF_TOKEN
# )

# Load model with FP16
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map={"": device},
    token=HF_TOKEN
)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    use_rslora=True,
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # potentially add mlp here if needed
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_epochs = 3
batch_size = 1

# Prepare model and optimizer with accelerate
model, optimizer = accelerator.prepare(model, optimizer)

print("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"- {name}")
model.print_trainable_parameters()

# Load dataset
df = pd.read_csv(DATA_PATH)
questions = df["Question"].tolist()
hedged_answers = df["Hedging_Answer"].tolist()
confident_answers = df["Confident_Answer"].tolist()

# Get token IDs for ratings 1-5
rating_tokens = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]
print(f"Rating token IDs: {rating_tokens}")

loss_history = defaultdict(list)

def plot_loss_history(history):
    """Plot loss components over epochs"""
    plt.figure(figsize=(10, 6))
    for loss_name, values in history.items():
        plt.plot(values, label=loss_name)
    plt.xlabel('Batch')
    plt.ylabel('Loss Value')
    plt.title(f'Loss Components')
    plt.legend()
    plt.grid(True)
    plt.close()

for epoch in tqdm(range(num_epochs), desc="Epoch"):
    batch_pbar = tqdm(range(0, len(questions), batch_size), desc="Batch", leave=False)
    model.train()

    for i in batch_pbar:
        batch_questions = questions[i:i + batch_size]
        batch_hedged = hedged_answers[i:i + batch_size]
        batch_confident = confident_answers[i:i + batch_size]

        # Prepare inputs
        hedged_prompts = [EVALUATION_PROMPT.format(categories="\n".join(EVALUATION_CATEGORIES), question=q, response=r) for q, r in zip(batch_questions, batch_hedged)]
        confident_prompts = [EVALUATION_PROMPT.format(categories="\n".join(EVALUATION_CATEGORIES), question=q, response=r) for q, r in zip(batch_questions, batch_confident)]

        hedged_inputs = tokenizer(hedged_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        confident_inputs = tokenizer(confident_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        # Use accelerator's gradient accumulation context
        with accelerator.accumulate(model):
            # Forward pass
            hedged_outputs = model(**hedged_inputs, output_hidden_states=True)
            confident_outputs = model(**confident_inputs, output_hidden_states=True)

            # Extract hidden states and logits (cast to float32 for loss)
            hedged_hidden = hedged_outputs.hidden_states[-1][:, -1, :].to(torch.float32)
            confident_hidden = confident_outputs.hidden_states[-1][:, -1, :].to(torch.float32)
            hedged_logits = hedged_outputs.logits[:, -1, :].to(torch.float32)
            confident_logits = confident_outputs.logits[:, -1, :].to(torch.float32)

            # Compute probabilities and scores in float32
            hedged_probs = F.softmax(hedged_logits[:, rating_tokens], dim=-1)
            confident_probs = F.softmax(confident_logits[:, rating_tokens], dim=-1)
            vocab_scores = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device=device)  # Float32 for loss
            hedged_scores = torch.sum(hedged_probs * vocab_scores, dim=-1)
            confident_scores = torch.sum(confident_probs * vocab_scores, dim=-1)

            # Loss components in float32
            score_loss = torch.mean((hedged_scores - confident_scores) ** 2)
            dist_loss = torch.mean(F.kl_div(F.log_softmax(hedged_logits[:, rating_tokens], dim=-1),
                                        F.softmax(confident_logits[:, rating_tokens], dim=-1), reduction='batchmean'))
            hidden_loss = torch.mean((hedged_hidden - confident_hidden) ** 2)
            reg_loss = 0.1 * torch.mean(hedged_scores ** 2 + confident_scores ** 2)
            l_c, l_d, l_h, l_r = 0.5, 0.5, 0.2, 0.1
            total_loss = l_c * score_loss + l_d * dist_loss + l_h * hidden_loss + l_r * reg_loss

            # Backward pass (accelerate handles accumulation)
            accelerator.backward(total_loss)
            
            # Update optimizer
            optimizer.step()
            optimizer.zero_grad()
        if accelerator.is_main_process:
            loss_history['Total'].append(total_loss.item())
            loss_history['Score'].append(l_c * score_loss.item())
            loss_history['Dist'].append(l_d * dist_loss.item())
            loss_history['Hidden'].append(l_h * hidden_loss.item())
            loss_history['Reg'].append(l_r * reg_loss.item())

        # Report per-batch loss
        batch_pbar.set_postfix({
            "Total": f"{total_loss.item():.4f}",
            "Score": f"{score_loss.item():.4f}",
            "Dist": f"{dist_loss.item():.4f}",
            "Hidden": f"{hidden_loss.item():.4f}",
            "Reg": f"{reg_loss.item():.4f}"
        })

    if accelerator.is_main_process:  # Only main process handles cleanup
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    gc.collect()

plot_loss_history(loss_history)

# Save the fine-tuned model (unwrap from accelerate)
model = accelerator.unwrap_model(model)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model saved to {SAVE_DIR}")