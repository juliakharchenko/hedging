import pandas as pd
import os

# super simple iteration script to show one final reaosning at a time for qual coding purposes
DATA_FILE_NAME = "interview_scores_llama-3.3-70B"
df = pd.read_csv(f"../results/{DATA_FILE_NAME}.csv")
for idx, row in df.iterrows():
    os.system("cls")
    print(f"Q{idx}, with {row['hedged_or_confident']} responses, final answer = {row['final_answer']}")
    print(row["final_answer_reasoning"])
    a = input()

    # just hit enter to go next lol
    if a:
        break