import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set the directory where your fine-tuned model is saved
SAVE_DIR = "fine_tuned_meta-llama_Llama-3.2-1B-Instruct"

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


# Define your prompt
prompt = f"""You are a hiring manager evaluating a candidate's response to an interview question. You want to hire the best candidate for the job and are going to think critically and rate the candidate's response.

Evaluate the interview response with the following categories in mind:
{"\n".join(EVALUATION_CATEGORIES)}

You will rate the interview response on a scale from 1 to 5 based on the content of the response, using the following scale:
5 - Far Exceeds Requirements: Perfect answer. Demonstrates competency accurately, consistently, and independently. All points relevant. The answer includes only good and relavant examples.
4 - Exceeds Requirements: Demonstrates competency accurately and consistently in most situations with minimal guidance. The answer would benift from examples and many good examples provided.
3 - Meets Requirements: Demonstrates competency accurately and consistently on familiar procedures and needs supervisor guidance for new skills. The answer would benift from examples and some good examples provided.
2 - Below Requirements: Demonstrates competency inconsistently, even with repeated instruction or guidance. Although the answer would benift from examples, few good examples provided.
1 - Significant Gap: Fails to demonstrate competency regardless of guidance provided. Although the answer would benift from examples, no good examples provided.

It is extremely important that as a response you provide a clear, detailed, and conscise explanation along with the rating.
Provide your response in the following format (replace <your rating> and <your reasoning> with the values you decide):
```
Reasoning: <your reasoning>
Rating: <your rating>
```

A couple examples of responses are:
```
Reasoning: The candidate provided a clear and concise response to the question, but lacked depth in the examples they provided.
Rating: 3
```
```
Reasoning: The candidate failed to provide a clear or relevant response to the question.
Rating: 1
```
```
Reasoning: The candidate provided a detailed and thorough response to the question, with relevant examples.
Rating: 5
```

Context:
The question that was asked was: "Tell me about yourself."
The interview response was: "I have five years of experience in software engineering, specializing in backend development and scalable system design."
"""

# Tokenize the prompt and move inputs to device
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate a response
output_ids = model.generate(**inputs, max_length=833, do_sample=True, top_k=50)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated Text:\n", output_text)