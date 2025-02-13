import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads experiment results from a CSV file.
    Assumes the file has headers such as:
    interview_round, hedged_or_confident, question_1, question_1_category, question_1_response,
    question_1_score, question_1_score_reasoning, ... , final_answer, final_answer_reasoning
    """
    df = pd.read_csv(file_path, encoding='utf-8')
    return df

def compute_average_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts question score columns to numeric values and computes an average score
    per interview session.
    """
    # Identify score columns for the questions (assumes they include "_score" in their names but not the final answer)
    score_columns = [col for col in df.columns if col.startswith("question_") and col.endswith("_score")]
    
    # Convert to numeric, coercing errors to NaN
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute the average score for each row
    df["avg_score"] = df[score_columns].mean(axis=1)
    return df

def analyze_scores(df: pd.DataFrame):
    """
    Separates the responses into Hedged and Confident groups,
    performs a t-test on the average scores, and visualizes the differences.
    """
    # Normalize the hedged/confident labels to lowercase for consistency
    df["response_type"] = df["hedged_or_confident"].str.lower()
    
    hedged = df[df["response_type"] == "hedged"]["avg_score"].dropna()
    confident = df[df["response_type"] == "confident"]["avg_score"].dropna()

    # Perform an independent t-test
    t_stat, p_value = stats.ttest_ind(hedged, confident, equal_var=False)
    print(f"T-test results: t-statistic = {t_stat:.3f}, p-value = {p_value:.3e}")

    # Create a boxplot comparing the two groups
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="hedged_or_confident", y="avg_score", data=df, palette="Set2")
    plt.title("Comparison of Average Scores: Hedged vs Confident Responses")
    plt.xlabel("Response Type")
    plt.ylabel("Average Score")
    plt.tight_layout()
    plt.savefig("avg_score_boxplot.png")
    plt.show()

def additional_analysis(df: pd.DataFrame):
    """
    (Optional) Extend this function to add more detailed analyses.
    For example, analyze per-question differences or correlate final decisions with scores.
    """
    # Example: Count final decision outcomes by response type
    if "final_answer" in df.columns:
        decision_counts = df.groupby(["hedged_or_confident", "final_answer"]).size().unstack(fill_value=0)
        print("\nFinal Decision Counts by Response Type:")
        print(decision_counts)
        
        # Plot a stacked bar chart
        decision_counts.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")
        plt.title("Final Decision Outcomes by Response Type")
        plt.xlabel("Response Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("final_decision_stacked_bar.png")
        plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <path_to_csv>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    df = load_data(file_path)
    df = compute_average_scores(df)
    analyze_scores(df)
    additional_analysis(df)

if __name__ == "__main__":
    main()
