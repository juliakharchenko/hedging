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

# TODO: experiment with different categories
# Define evaluation categories (can be extended or parameterized)
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
    for session in range(num_sessions):
        selected_questions = random.sample(questions, 10)
        for response_type in ['Hedged', 'Confident']:
            responses_summary = []
            row = [session + 1, response_type]
            for i, (question, hedged, confident) in enumerate(selected_questions):
                category = EVALUATION_CATEGORIES[i % len(EVALUATION_CATEGORIES)] # TODO: Why do we do this?
                response = hedged if response_type == 'Hedged' else confident
                score, reasoning = rate_response_with_llm(response, category, tokenizer, model)
                row.extend([question, category, response, score, reasoning])
                responses_summary.append(f"Category: {category}, Score: {score}, Reasoning: {reasoning}")
            final_decision = determine_outcome_with_llm("\n".join(responses_summary), tokenizer, model)
            advance_or_not = determine_advance_or_not(final_decision, tokenizer, model)
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
