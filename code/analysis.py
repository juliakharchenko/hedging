import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

##############################
# Data Loading & Preprocessing
##############################

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads experiment results from a CSV file.
    Expects headers like: interview_round, hedged_or_confident,
    question_1, question_1_category, question_1_response,
    question_1_score, question_1_score_reasoning, ... , final_answer, final_answer_reasoning.
    """
    df = pd.read_csv(file_path, encoding='utf-8')
    # Store file name (without extension) for figure naming and report output
    df.file_name = os.path.splitext(os.path.basename(file_path))[0]
    return df

def compute_average_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts question score columns to numeric values and computes an average score per interview session.
    """
    score_columns = [col for col in df.columns if col.startswith("question_") and col.endswith("_score")]
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df["avg_score"] = df[score_columns].mean(axis=1)
    return df

##############################
# Plotting Functions
##############################

def plot_boxplot_avg_scores(df: pd.DataFrame, out_dir: str):
    """Creates and saves a boxplot comparing average scores for Hedged vs Confident responses."""
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x="hedged_or_confident", hue="hedged_or_confident", y="avg_score", data=df)
    ax.set_title("Average Scores by Response Type", fontsize=16)
    ax.set_xlabel("Response Type", fontsize=14)
    ax.set_ylabel("Average Score", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"avg_score_boxplot_{df.file_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_stacked_bar_decisions(decision_counts: pd.DataFrame, file_name: str, out_dir: str):
    """Creates and saves a stacked bar chart for final decision outcomes using a neutral color scheme."""
    plt.figure(figsize=(10, 8))
    ax = decision_counts.plot(kind="bar", stacked=True, figsize=(10, 8), colormap="coolwarm")
    ax.set_title("Final Decision Outcomes by Response Type", fontsize=16)
    ax.set_xlabel("Response Type", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    plt.xticks(rotation=0)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"final_decision_stacked_bar_{file_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_reasoning_length_bar(reasoning_stats: pd.DataFrame, file_name: str, out_dir: str):
    """Creates and saves a bar chart for average reasoning length by final decision group."""
    plt.figure(figsize=(8, 6))
    ax = reasoning_stats["mean"].plot(kind="bar", yerr=reasoning_stats["std"], capsize=5,
                                      color="skyblue", edgecolor="black")
    ax.set_title("Average Reasoning Length by Final Decision", fontsize=16)
    ax.set_xlabel("Final Decision", fontsize=14)
    ax.set_ylabel("Average Reasoning Length", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"avg_reasoning_length_bar_{file_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

##############################
# Report Utility
##############################

def save_report(report: str, file_name: str, out_dir: str):
    """Saves the provided report text to a file in the specified output directory."""
    out_path = os.path.join(out_dir, file_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

##############################
# Statistical Analysis Functions
##############################

def analyze_scores(df: pd.DataFrame, out_dir: str) -> str:
    """
    Separates responses into Hedged and Confident groups,
    performs a t-test on the average scores, visualizes the differences,
    and returns a string report of the statistics.
    """
    report = []
    df["response_type"] = df["hedged_or_confident"].str.lower()
    hedged = df[df["response_type"] == "hedged"]["avg_score"].dropna()
    confident = df[df["response_type"] == "confident"]["avg_score"].dropna()

    # Perform an independent t-test
    t_stat, p_value = stats.ttest_ind(hedged, confident, equal_var=False)
    report.append("T-test for Average Scores (Hedged vs Confident):")
    report.append(f"  t-statistic: {t_stat:.3f}")
    report.append(f"  p-value: {p_value:.3e}\n")

    # Plot and save the boxplot
    plot_boxplot_avg_scores(df, out_dir)
    return "\n".join(report)

def analyze_final_decisions(df: pd.DataFrame, out_dir: str) -> str:
    """
    Analyzes final decision outcomes by response type, plots the results,
    and returns a string report of the statistics.
    """
    report = []
    if "final_answer" in df.columns:
        df["final_answer"] = df["final_answer"].str.lower()
        df["final_answer_group"] = np.where(df["final_answer"].str.contains("not"), "reject", "accept")
        decision_counts = df.groupby(["response_type", "final_answer_group"]).size().unstack(fill_value=0)
        report.append("Final Decision Counts by Response Type:")
        report.append(decision_counts.to_string())
        report.append("")
        plot_stacked_bar_decisions(decision_counts, df.file_name, out_dir)

        # Average scores by decision group
        score_stats = df.groupby("final_answer_group")["avg_score"].agg(["mean", "std", "count"])
        report.append("Average Score by Final Decision Group:")
        report.append(score_stats.to_string())
        report.append("")
    return "\n".join(report)

def analyze_reasoning_length(df: pd.DataFrame, out_dir: str) -> str:
    """
    Analyzes reasoning length, correlates it with average scores,
    and returns a string report of the statistics along with plotting.
    """
    report = []
    if "final_answer_reasoning" in df.columns:
        df["reasoning_length"] = df["final_answer_reasoning"].str.len()
        if "final_answer_group" not in df.columns:
            # Ensure final_answer_group exists
            df["final_answer"] = df["final_answer"].str.lower()
            df["final_answer_group"] = np.where(df["final_answer"].str.contains("not"), "reject", "accept")
        reasoning_stats = df.groupby("final_answer_group")["reasoning_length"].agg(["mean", "std", "count"])
        report.append("Average Reasoning Length by Final Decision:")
        report.append(reasoning_stats.to_string())
        report.append("")
        plot_reasoning_length_bar(reasoning_stats, df.file_name, out_dir)

        # Correlation between reasoning length and average score
        corr = df[["avg_score", "reasoning_length"]].dropna().corr()
        report.append("Correlation between Average Score and Reasoning Length:")
        report.append(corr.to_string())
        report.append("")
    return "\n".join(report)

def additional_analysis(df: pd.DataFrame, out_dir: str) -> str:
    """Runs additional analyses and returns a combined report string."""
    report_parts = []
    report_parts.append(analyze_final_decisions(df, out_dir))
    report_parts.append(analyze_reasoning_length(df, out_dir))
    return "\n".join(report_parts)

##############################
# Utility Functions
##############################

def create_output_dir(directory: str = "figs") -> str:
    """Creates output directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

##############################
# Main Routine
##############################

def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <path_to_csv_directory>")
        sys.exit(1)

    # Configure Seaborn and Matplotlib for better readability
    plt.ioff()
    sns.set_palette("Set2")
    sns.set_style("whitegrid")
    sns.set_context("talk")

    out_dir = create_output_dir("figs")
    input_dir = sys.argv[1]
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the provided directory.")
        sys.exit(1)

    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        report = []
        report.append(f"Analysis Report for {file_path}\n{'='*60}\n")
        try:
            df = load_data(file_path)
            df = compute_average_scores(df)
            report.append(analyze_scores(df, out_dir))
            report.append(additional_analysis(df, out_dir))
        except Exception as e:
            report.append(f"Error analyzing {file_path}: {e}")
        # Save the report to a file
        report_text = "\n".join(report)
        report_file_name = f"analysis_report_{os.path.splitext(file)[0]}.txt"
        save_report(report_text, report_file_name, out_dir)
        print(f"Saved analysis report to {os.path.join(out_dir, report_file_name)}")

if __name__ == "__main__":
    main()
