import os
import random
from typing import List, Tuple
from data_utils import load_questions, save_results
from evaluator import (
    rate_response_with_llm,
    determine_outcome_with_llm,
    determine_advance_or_not
)
from transformers import PreTrainedTokenizer, PreTrainedModel
from tqdm import tqdm

def analyze_candidate_performance(
    questions: List[Tuple[str, str, str]],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    num_sessions: int = 20
) -> List:
    """
    Conducts interview sessions, evaluates answers, and determines overall candidate performance.
    """
    results = []
    for session in tqdm(range(num_sessions)):
        selected_questions = random.sample(questions, 10)
        for response_type in ['Hedged', 'Confident']:
            responses_summary = []
            row = [session + 1, response_type]
            for question, hedged, confident in tqdm(selected_questions, desc=response_type):
                response = hedged if response_type == 'Hedged' else confident
                score, reasoning = rate_response_with_llm(question, response, tokenizer, model)
                row.extend([question, response, score, reasoning])
                responses_summary.append(f"Question: {question}\nScore (out of 5) of interviewee response with reasoning:\n{reasoning}")
            final_decision = determine_outcome_with_llm("\n-------------\n".join(responses_summary), tokenizer, model)
            advance_or_not = determine_advance_or_not(final_decision)
            row.extend([advance_or_not, final_decision])
            results.append(row)
    return results

def run_experiment_for_model(
    model_id: str,
    questions_file: str,
    results_folder: str,
    tokenizer,
    model
) -> None:
    """
    Runs the experiment for a single model and saves the results.
    """
    questions = load_questions(questions_file)
    results = analyze_candidate_performance(questions, tokenizer, model)
    
    # Ensure the results_folder exists
    os.makedirs(results_folder, exist_ok=True)

    # Save results directly to results_folder with a modified filename
    output_file = os.path.join(results_folder, f"interview_scores_{model_id.replace('/', '_')}.csv")
    save_results(results, output_file)
    print(f"Results saved to {output_file}")
