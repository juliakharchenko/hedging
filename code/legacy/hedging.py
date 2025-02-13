import os

os.environ["HF_TOKEN"] = # put your token
import random
import csv
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model_id = "meta-llama/Llama-3.1-8B-Instruct" # TODO: Change to different models
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda', quantization_config=bnb_config)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def load_questions(filename):
    """Loads questions from a CSV file with columns: question, hedged answer, confident answer."""
    questions = []
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) == 3:
                question, hedged, confident = row
                questions.append((question.strip(), hedged.strip(), confident.strip()))
    return questions

def rate_response_with_llm(response, category):
    """Uses LLaMA to rate a response on a scale of 1-5 with reasoning based on a structured interview rubric."""
    messages = [{"role": "user",
        "content": f"""
          Evaluate the following interview response under the category: {category}.
          Rate it on a scale from 1 to 5 based on clarity, relevance, and depth.
          Use the following scale:
          5 - Far Exceeds Requirements: Perfect answer. Demonstrates competency accurately, consistently, and independently. All points relevant. All good examples.
          4 - Exceeds Requirements: Demonstrates competency accurately and consistently in most situations with minimal guidance. Many good examples.
          3 - Meets Requirements: Demonstrates competency accurately and consistently on familiar procedures and needs supervisor guidance for new skills. Some good examples.
          2 - Below Requirements: Demonstrates competency inconsistently, even with repeated instruction or guidance. Few good examples.
          1 - Significant Gap: Fails to demonstrate competency regardless of guidance provided. No good examples.
          Response: "{response}"
          Provide a numerical rating and a brief reasoning.
          """
                }]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    llm_response = tokenizer.decode(response, skip_special_tokens=True)
    print("response = " + llm_response)
    score = determine_score(llm_response)

    print("score = " + score)
    return score, llm_response

def determine_score(response):
    messages = [{"role": "user",
        "content": f"""
          Given the following responses, determine what score the response indicates, on a scale from 1 to 5.
          Return the number only, and no extra explanation.
          Example
          Response: I would give this candidate a score of 3, because it was average.
          Score: 3

          Now, your turn.
          Response: {response}
          Score:
      """
                }]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    score = outputs[0][input_ids.shape[-1]:]
    llm_score = tokenizer.decode(score, skip_special_tokens=True)
    return llm_score

def determine_outcome_with_llm(responses):
    """Uses LLaMA to determine whether the candidate should advance based on all responses."""
    messages = [{
        "role": "user",
        "content": f"""
    Based on the following interview responses and their ratings, determine whether the candidate should:
    - Advance
    - Advance with reservations
    - Do not advance
    Provide a final decision and a brief reasoning.

    Responses:
    {responses}
    """
    }]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    answer = outputs[0][input_ids.shape[-1]:]
    llm_answer = tokenizer.decode(answer, skip_special_tokens=True)
    return llm_answer

def analyze_candidate_performance(questions, num_sessions=20):
    """Conducts interview sessions, evaluates answers, and determines overall candidate performance."""
    evaluation_categories = [
        "Educational Background: Does the candidate have the appropriate educational qualifications or training for this position?",
        "Job/Organizational Fit: Has the candidate acquired similar skills or qualifications through past work experiences?",
        "Problem Solving: Did the candidate display the ability to react to a situation anddevise a strategy?",
        "Verbal Communication: How were the candidate’s communication skills during the interview?",
        "Candidate Interest: How much interest did the candidate show in the position and the organization?",
        "Knowledge of Organization: Did the candidate research the organization prior to the interview?",
        "Teambuilding/Interpersonal Skills: Did the candidate demonstrate, through their answers, good teambuilding/interpersonal skills?",
        "Initiative: – Did the candidate demonstrate, through their answers, a high degree of initiative?",
        "Time Management: Did the candidate demonstrate, through their answers, good time management skills?",
        "Attention to Detail: Was the candidate able to give provide examples of detail in their previous work experience?"
    ]
    results = []
    for _ in range(num_sessions):
        session_scores = {'Hedged': [], 'Confident': []}
        responses_summary = []
        selected_questions = random.sample(questions, 10)
        for i, (question, hedged, confident) in enumerate(selected_questions):
            category = evaluation_categories[i % len(evaluation_categories)]
            hedged_score, hedged_reason = rate_response_with_llm(hedged, category)
            confident_score, confident_reason = rate_response_with_llm(confident, category)
            session_scores['Hedged_Reason'] = hedged_reason
            session_scores['Confident_Reason'] = confident_reason
            session_scores['Hedged'].append(hedged_score)
            session_scores['Confident'].append(confident_score)
            responses_summary.append(f"Category: {category}, Hedged Score: {hedged_score}, Reason: {hedged_reason}")
            responses_summary.append(f"Category: {category}, Confident Score: {confident_score}, Reason: {confident_reason}")
            results.append([question, category, 'Hedged', hedged_score, hedged_reason])
            results.append([question, category, 'Confident', confident_score, confident_reason])

        final_decision_hedged = determine_outcome_with_llm("\n".join(responses_summary))
        final_decision_confident = determine_outcome_with_llm("\n".join(responses_summary))

        results.append(["Final Decision", "", "Hedged", final_decision_hedged])
        results.append(["Final Decision", "", "Confident", final_decision_confident])

    return results

def save_results(results, filename):
    """Saves interview results to a CSV file."""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Question", "Category", "Response Type", "Score", "Reasoning"])
        writer.writerows(results)

# Load questions and execute interview analysis
questions_file = "../data/hedging_questions.csv"
questions = load_questions(questions_file)
interview_results = analyze_candidate_performance(questions)
save_results(interview_results, "interview_scores_[model_name].csv") # TODO: Change model name!
