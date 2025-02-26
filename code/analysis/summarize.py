import os
import sys
import re
import pandas as pd
from typing import Dict, Optional, List

def summarize_csv(file_path: str) -> Dict[str, Optional[float]]:
    """
    Summarizes the CSV file by computing relevant statistics.
    
    Statistics computed:
      - file_name: Shortened file name (removes "interview_scores_" prefix and ".csv" suffix).
      - num_records: Number of rows in the CSV.
      - overall_avg_score: Mean of the average score per row computed from all question_{x}_score columns.
      - overall_std_score: Standard deviation of the average score per row.
      - Per-question statistics for each question_{x}_score column (mean, std, count).
      - count_accept: Total number of rows with final_answer indicating acceptance.
      - count_reject: Total number of rows with final_answer indicating rejection.
      - hedged_avg_score, hedged_std_score, hedged_count: Statistics for rows with hedged responses.
      - confident_avg_score, confident_std_score, confident_count: Statistics for rows with confident responses.
      - hedged_accept_count, hedged_reject_count, confident_accept_count, confident_reject_count:
        Breakdown counts for final decisions within each response type.
    
    Grouping:
      - final_answer_group: 'reject' if final_answer contains "not", else 'accept'.
      - response_type: derived from hedged_or_confident, normalized to lowercase.
    
    Args:
        file_path: Path to the CSV file.
    
    Returns:
        A dictionary containing the summary statistics.
    """
    df = pd.read_csv(file_path, encoding="utf-8")
    
    # Shorten file name
    fname = os.path.basename(file_path)
    fname = fname.replace("interview_scores_", "")
    if fname.endswith(".csv"):
        fname = fname[:-4]
    
    summary: Dict[str, Optional[float]] = {}
    summary["file_name"] = fname
    summary["num_records"] = len(df)
    
    # Process question score columns (e.g., question_1_score, question_2_score, ...)
    score_cols = [col for col in df.columns if re.match(r'^question_\d+_score$', col)]
    if score_cols:
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Compute an average score per row
        df["avg_score"] = df[score_cols].mean(axis=1)
        summary["overall_avg_score"] = df["avg_score"].mean()
        summary["overall_std_score"] = df["avg_score"].std()
        
        # Compute per-question statistics
        for col in score_cols:
            summary[f"{col}_mean"] = df[col].mean()
            summary[f"{col}_std"] = df[col].std()
            summary[f"{col}_count"] = df[col].notnull().sum()
    else:
        summary["overall_avg_score"] = None
        summary["overall_std_score"] = None

    # Process final_answer column for accept/reject grouping
    if "final_answer" in df.columns:
        df["final_answer"] = df["final_answer"].astype(str).str.strip().str.lower()
        df["final_answer_group"] = df["final_answer"].apply(lambda x: "reject" if "not" in x else "accept")
        summary["count_accept"] = (df["final_answer_group"] == "accept").sum()
        summary["count_reject"] = (df["final_answer_group"] == "reject").sum()
    else:
        summary["count_accept"] = None
        summary["count_reject"] = None

    # Process hedged_or_confident column for grouping statistics and breakdowns
    if "hedged_or_confident" in df.columns and score_cols:
        df["response_type"] = df["hedged_or_confident"].astype(str).str.strip().str.lower()
        grouped = df.groupby("response_type")["avg_score"].agg(['mean', 'std', 'count'])
        # For hedged responses
        if "hedged" in grouped.index:
            summary["hedged_avg_score"] = grouped.loc["hedged", "mean"]
            summary["hedged_std_score"] = grouped.loc["hedged", "std"]
            summary["hedged_count"] = grouped.loc["hedged", "count"]
        else:
            summary["hedged_avg_score"] = None
            summary["hedged_std_score"] = None
            summary["hedged_count"] = 0
        # For confident responses
        if "confident" in grouped.index:
            summary["confident_avg_score"] = grouped.loc["confident", "mean"]
            summary["confident_std_score"] = grouped.loc["confident", "std"]
            summary["confident_count"] = grouped.loc["confident", "count"]
        else:
            summary["confident_avg_score"] = None
            summary["confident_std_score"] = None
            summary["confident_count"] = 0
        
        # Breakdown counts for accept/reject by response type
        summary["hedged_accept_count"] = df[(df["response_type"] == "hedged") & (df["final_answer_group"] == "accept")].shape[0]
        summary["hedged_reject_count"] = df[(df["response_type"] == "hedged") & (df["final_answer_group"] == "reject")].shape[0]
        summary["confident_accept_count"] = df[(df["response_type"] == "confident") & (df["final_answer_group"] == "accept")].shape[0]
        summary["confident_reject_count"] = df[(df["response_type"] == "confident") & (df["final_answer_group"] == "reject")].shape[0]
    else:
        summary["hedged_avg_score"] = None
        summary["hedged_std_score"] = None
        summary["hedged_count"] = None
        summary["confident_avg_score"] = None
        summary["confident_std_score"] = None
        summary["confident_count"] = None
        summary["hedged_accept_count"] = None
        summary["hedged_reject_count"] = None
        summary["confident_accept_count"] = None
        summary["confident_reject_count"] = None

    return summary

def summarize_directory(input_dir: str, output_csv: str = "summary.csv") -> None:
    """
    Processes all CSV files in the input directory, computes summary statistics,
    and writes them to a single summary CSV file.
    
    Args:
        input_dir: Directory containing CSV files.
        output_csv: File path for the summary CSV.
    """
    if not os.path.isdir(input_dir):
        print(f"Input directory '{input_dir}' does not exist or is not a directory.")
        sys.exit(1)
    
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        print("No CSV files found in the provided directory.")
        sys.exit(1)
    
    summaries: List[Dict[str, Optional[float]]] = []
    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        try:
            summary = summarize_csv(file_path)
            summaries.append(summary)
            print(f"Processed {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Summary CSV saved to: {output_csv}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize.py <path_to_csv_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    summarize_directory(input_dir)

if __name__ == "__main__":
    main()
