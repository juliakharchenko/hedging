import argparse
import os
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configuration parameters
ORIGINAL_DATA_PATH = "../../data/hedging_questions.csv"
OUTPUT_DATA_PATH = "generated_training_data.csv"
HF_MODEL_NAME = "openai-community/gpt2"  # Change to any Hugging Face text-generation model
NUM_FEW_SHOT = 5     # Number of few-shot examples to include in the prompt

def load_original_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def get_few_shot_examples(df: pd.DataFrame, num_examples: int) -> str:
    examples = []
    sample_df = df.sample(n=num_examples, random_state=42)
    for _, row in sample_df.iterrows():
        example = (
            f"Question: {row['Question']}\n"
            f"Hedging Answer: {row['Hedging_Answer']}\n"
            f"Confident Answer: {row['Confident_Answer']}"
        )
        examples.append(example)
    return "\n\n".join(examples)

def build_prompt(few_shot_examples: str) -> str:
    prompt = (
        "You are tasked with generating new question-answer triplets for a dataset of interview questions. Hedged answers indicate usage of hedging throughout language, and confident answers indicate confident language. The content of the hedged answers and confident answers, and the knowledge of the speaker, is exactly the same - the only difference between these answers is the language used.\n"
        "Below are some examples of interview question-answer triplets:\n\n"
        f"{few_shot_examples}\n\n"
        "Now its your turn, generate a new interview question and its corresponding answers ensuring that it is in the following format (replace everything in and including the <>):\n"
        "Question: <new question text>\n"
        "Hedging Answer: <new hedging answer text>\n"
        "Confident Answer: <new confident answer text>\n"
    )
    return prompt

def parse_triplet(text: str) -> dict:
    pattern = r"Question:\s*(.*?)\s*Hedging Answer:\s*(.*?)\s*Confident Answer:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return {
            "Question": match.group(1).strip(),
            "Hedging_Answer": match.group(2).strip(),
            "Confident_Answer": match.group(3).strip()
        }
    return None

def main(num_new: int):
    # Load original data and prepare few-shot examples.
    df_orig = load_original_data(ORIGINAL_DATA_PATH)
    few_shot = get_few_shot_examples(df_orig, NUM_FEW_SHOT)
    prompt = build_prompt(few_shot)

    # Load model and tokenizer; build text-generation pipeline.
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print("Generating new triplets...")
    outputs = generator(prompt, num_return_sequences=num_new, max_new_tokens=150)
    triplets = []
    for i, out in enumerate(outputs, 1):
        # Remove the prompt from the generated text if it's included.
        text = out["generated_text"]
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        triplet = parse_triplet(text)
        if triplet:
            triplets.append(triplet)
            print(f"[{i}/{num_new}] Successfully parsed triplet.")
        else:
            print(f"[{i}/{num_new}] Failed to parse triplet.")

    # Save the new triplets to CSV.
    pd.DataFrame(triplets).to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"Saved {len(triplets)} triplets to {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate new question-answer triplets using a Hugging Face model.")
    parser.add_argument("--num", type=int, default=10, help="Number of new triplets to generate")
    args = parser.parse_args()
    main(args.num)