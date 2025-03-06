import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name: str):
    """
    Loads the tokenizer and model from Hugging Face.
    Adjust torch_dtype and device_map as needed for your hardware.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use fp16 for efficiency on GPU
        device_map="auto"
    )
    return tokenizer, model

def generate_hedged_answer(model, tokenizer, original_answer: str, prompt_template: str, gen_params: dict):
    """
    Formats the prompt, runs the model, and extracts the hedged answer.
    """
    # Prepare the prompt using the provided template.
    prompt = prompt_template.format(original_answer=original_answer)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Generate output using the provided generation parameters.
    output_ids = model.generate(input_ids, **gen_params)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract text following the "Hedged Answer:" marker.
    if "Hedged Answer:" in output_text:
        hedged_text = output_text.split("Hedged Answer:")[-1].strip()
    else:
        hedged_text = output_text.strip()
    return hedged_text

def main():
    parser = argparse.ArgumentParser(
        description="Augment CSV with hedged answers using a Llama 70B model from Hugging Face."
    )
    parser.add_argument(
        "--csv", type=str, default="data.csv",
        help="Path to the input CSV file (must contain an 'answer' column)."
    )
    parser.add_argument(
        "--output", type=str, default="data_augmented.csv",
        help="Path to save the augmented CSV file."
    )
    parser.add_argument(
        "--model", type=str, default="decapoda-research/llama-70b",
        help="Hugging Face model identifier for Llama 70B."
    )
    args = parser.parse_args()

    # Load the CSV
    df = pd.read_csv(args.csv)
    if 'answer' not in df.columns:
        raise ValueError("CSV must contain a column named 'answer'.")

    # Load the model and tokenizer
    print("Loading model...")
    tokenizer, model = load_model(args.model)

    # Define a prompt template optimized for generating hedged language.
    prompt_template = (
        "You are an expert language model specialized in rephrasing text with hedged language. "
        "Please rephrase the following answer using cautious and tentative expressions (e.g., 'it seems', 'possibly', 'likely', etc.) while preserving the original meaning. "
        "Do not add any extra commentary or modify the content beyond rephrasing. "
        "Only output the rephrased answer.\n\n"
        "Original Answer: \"{original_answer}\"\n\n"
        "Hedged Answer:"
    )

    # Set generation parameters – feel free to adjust these based on your results.
    gen_params = {
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
        "num_return_sequences": 1,
    }

    # Process each row and generate the hedged answer.
    hedged_answers = []
    print("Processing CSV rows...")
    for idx, row in df.iterrows():
        orig_ans = row['answer']
        hedged_ans = generate_hedged_answer(model, tokenizer, orig_ans, prompt_template, gen_params)
        hedged_answers.append(hedged_ans)
        print(f"Processed row {idx + 1} of {len(df)}")

    # Add the hedged answer as a new column and save the augmented CSV.
    df["hedged_answer"] = hedged_answers
    df.to_csv(args.output, index=False)
    print(f"Augmented CSV saved to {args.output}")

if __name__ == "__main__":
    main()
