import os
from model_loader import load_model
from experiment import run_experiment_for_model

def main():
    # List of model IDs you want to experiment on
    models_to_run = [
        "meta-llama/Llama-3.1-8B-Instruct",
        # TODO: Add additional model IDs here as needed
    ]
    questions_file = "../data/hedging_questions.csv"
    results_folder = "../results"
    
    # Ensure HF_TOKEN is set in the environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    for model_id in models_to_run:
        print(f"Running experiment for model: {model_id}")
        tokenizer, model, _ = load_model(model_id, use_8bit=True)
        run_experiment_for_model(model_id, questions_file, results_folder, tokenizer, model)

if __name__ == "__main__":
    main()
