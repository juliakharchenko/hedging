import csv
from typing import List, Tuple

def load_questions(filename: str) -> List[Tuple[str, str, str]]:
    """
    Loads questions from a CSV file with columns: question, hedged answer, confident answer.
    """
    questions = []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            if len(row) == 3:
                question, hedged, confident = row
                questions.append((question.strip(), hedged.strip(), confident.strip()))
    return questions

def save_results(results: list, output_path: str) -> None:
    """
    Saves interview results to a CSV file at the given output_path.
    """
    header = ["interview_round", "hedged_or_confident"]
    for i in range(1, 11):
        header.extend([
            f"question_{i}", f"question_{i}_response",
            f"question_{i}_score", f"question_{i}_score_reasoning"
        ])
    header.extend(["final_answer", "final_answer_reasoning"])
    
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)
