import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set the directory where your fine-tuned model is saved
SAVE_DIR = "fine_tuned_openai-community_gpt2"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
model = AutoModelForCausalLM.from_pretrained(SAVE_DIR)

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model.to(device)

# Define your prompt
prompt = "What are your three favorite letters, only respond with the letters themselves?\n My favorite letters are:\n"

# Tokenize the prompt and move inputs to device
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate a response
output_ids = model.generate(**inputs, max_length=200, do_sample=True, top_k=50)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated Text:\n", output_text)