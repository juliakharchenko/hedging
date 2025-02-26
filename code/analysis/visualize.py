import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

# =============================================================================
# Mapping from shortened file name to a friendlier LLM name.
# Edit this dictionary as needed.
LLM_NAME_MAP = {
    "allenai_OLMoE-1B-7B-0125-Instruct": "OLMoE-1B-7B-0125",
    "CohereForAI_c4ai-command-r-plus-4bit": "Command-R-plus-4B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-R1-Qwen-1.5B",
    "google_gemma-2-2b-it": "Gemma-2-2B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "microsoft_phi-4": "Phi-4",
}

# =============================================================================
def shorten_filename(filename: str) -> str:
    """
    Shortens a filename by removing the prefix 'interview_scores_' and the '.csv' suffix.
    """
    name = filename
    if name.startswith("interview_scores_"):
        name = name[len("interview_scores_"):]
    if name.endswith(".csv"):
        name = name[:-4]
    return name

# =============================================================================
def load_and_process_file(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file and processes it:
      - Converts question score columns (question_#_score) to numeric.
      - Computes an overall score per row as the mean of question scores.
      - Normalizes the 'final_answer' column (assumes it is already normalized).
      - Normalizes the 'hedged_or_confident' column into 'response_type'.
      - Adds a column 'file' that maps to a friendlier LLM name if provided.
    """
    df = pd.read_csv(file_path, encoding="utf-8")
    
    # Shorten filename and map to LLM name if available
    short_name = shorten_filename(os.path.basename(file_path))
    llm_name = LLM_NAME_MAP.get(short_name, short_name)
    df["file"] = llm_name

    # Process question score columns
    score_cols = [col for col in df.columns if re.match(r'^question_\d+_score$', col)]
    if score_cols:
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df["overall_score"] = df[score_cols].mean(axis=1)
    else:
        df["overall_score"] = None

    # Process final_answer column (assumes already normalized)
    if "final_answer" in df.columns:
        df["final_answer"] = df["final_answer"].astype(str).str.strip().str.lower()
    else:
        df["final_answer"] = None

    # Process hedged_or_confident column
    if "hedged_or_confident" in df.columns:
        df["response_type"] = df["hedged_or_confident"].astype(str).str.strip().str.lower()
    else:
        df["response_type"] = None

    return df

# =============================================================================
def load_all_files(input_dir: str) -> pd.DataFrame:
    """
    Loads and processes all CSV files in the provided directory and returns a concatenated DataFrame.
    """
    all_files: List[pd.DataFrame] = []
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        print("No CSV files found in the provided directory.")
        sys.exit(1)
    
    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        try:
            df = load_and_process_file(file_path)
            all_files.append(df)
            print(f"Loaded and processed: {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if all_files:
        return pd.concat(all_files, ignore_index=True)
    else:
        print("No valid data loaded.")
        sys.exit(1)

# =============================================================================
def plot_violin_hedged_confident(df: pd.DataFrame) -> None:
    """
    Creates a violin plot comparing overall scores for hedged vs. confident responses.
    """
    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(
        x="response_type",
        y="overall_score",
        data=df,
        palette="Set2",
        order=["hedged", "confident"],
        inner="quartile"
    )
    ax.set_title("Overall Score Distribution: Hedged vs. Confident (LLMs)", fontsize=16)
    ax.set_xlabel("Response Type", fontsize=14)
    ax.set_ylabel("Overall Score", fontsize=14)
    plt.tight_layout()
    plt.show()

# =============================================================================
def plot_stacked_bar_accept_reject(df: pd.DataFrame) -> None:
    """
    Creates a stacked bar plot showing counts of each final decision (using the raw final_answer values)
    per LLM, with separate bars for hedged and confident responses.
    """
    import numpy as np  # ensure numpy is imported
    
    # Group by LLM, response_type, and final_answer
    counts = df.groupby(["file", "response_type", "final_answer"]).size().reset_index(name="count")
    
    # Sorted list of LLM names for x-axis ordering
    llms = sorted(df["file"].unique())
    
    # Define the response types order
    response_types_order = ["hedged", "confident"]
    
    # Get sorted unique final_answer categories from the data
    final_answers = sorted(df["final_answer"].dropna().unique())
    
    # Define a color mapping for final decision categories using a seaborn palette
    colors = sns.color_palette("Set2", len(final_answers))
    color_dict = dict(zip(final_answers, colors))
    
    # Create a pivot table with index = [file, response_type] and columns = final_answer
    pivot = counts.pivot_table(index=["file", "response_type"],
                               columns="final_answer",
                               values="count",
                               fill_value=0)
    
    # Separate the pivot data for hedged and confident responses (if available)
    try:
        pivot_hedged = pivot.xs("hedged", level="response_type", drop_level=False)
    except KeyError:
        pivot_hedged = None
    try:
        pivot_confident = pivot.xs("confident", level="response_type", drop_level=False)
    except KeyError:
        pivot_confident = None
    
    # Set up bar positions: one bar for hedged (left) and one for confident (right) per LLM.
    x = np.arange(len(llms))
    bar_width = 0.4  # width of each individual bar
    
    fig, ax = plt.subplots(figsize=(12, 7))
    added_labels = set()  # to avoid duplicate legend entries
    
    # Plot hedged bars (using hatching)
    if pivot_hedged is not None:
        # Ensure the data is in the same LLM order
        pivot_hedged = pivot_hedged.reindex(llms, level="file", fill_value=0)
        bottoms = np.zeros(len(llms))
        for decision in final_answers:
            # Get the counts for this decision; if missing, default to zeros
            values = pivot_hedged[decision].values if decision in pivot_hedged.columns else np.zeros(len(llms))
            pos = x - bar_width / 2  # shift left for hedged
            label = f"{decision.capitalize()} (Hedged)"
            if label in added_labels:
                label = None
            else:
                added_labels.add(label)
            ax.bar(pos, values, bar_width, bottom=bottoms,
                   color=color_dict[decision],
                   hatch="//",
                   edgecolor="black",
                   label=label)
            bottoms += values  # update the bottom for stacking
            
    # Plot confident bars (solid fill)
    if pivot_confident is not None:
        pivot_confident = pivot_confident.reindex(llms, level="file", fill_value=0)
        bottoms = np.zeros(len(llms))
        for decision in final_answers:
            values = pivot_confident[decision].values if decision in pivot_confident.columns else np.zeros(len(llms))
            pos = x + bar_width / 2  # shift right for confident
            label = f"{decision.capitalize()} (Confident)"
            if label in added_labels:
                label = None
            else:
                added_labels.add(label)
            ax.bar(pos, values, bar_width, bottom=bottoms,
                   color=color_dict[decision],
                   edgecolor="black",
                   label=label)
            bottoms += values
    
    ax.set_xticks(x)
    ax.set_xticklabels(llms, rotation=45, ha="right")
    ax.set_title("Final Decision Counts per LLM (by Response Type)", fontsize=16)
    ax.set_xlabel("LLM", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.legend(title="Final Decision", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    plt.show()


# =============================================================================
def plot_violin_overall_by_file(df: pd.DataFrame) -> None:
    """
    Creates a violin plot comparing the distribution of overall scores across LLMs.
    """
    sorted_llms = sorted(df["file"].unique())
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        x="file",
        y="overall_score",
        data=df,
        order=sorted_llms,
        palette="coolwarm",
        inner="quartile"
    )
    ax.set_title("Overall Score Distribution per LLM", fontsize=16)
    ax.set_xlabel("LLM", fontsize=14)
    ax.set_ylabel("Overall Score", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# =============================================================================
def additional_plots(df: pd.DataFrame) -> None:
    """
    Generates additional plots that might help answer the research question.
    For example, a scatter plot of overall score vs. reasoning length (if available).
    """
    if "final_answer_reasoning" in df.columns:
        df["reasoning_length"] = df["final_answer_reasoning"].astype(str).str.len()
        plt.figure(figsize=(8, 6))
        ax = sns.scatterplot(
            x="reasoning_length",
            y="overall_score",
            hue="response_type",
            data=df,
            palette="Set1"
        )
        ax.set_title("Overall Score vs. Reasoning Length (LLMs)", fontsize=16)
        ax.set_xlabel("Reasoning Length (characters)", fontsize=14)
        ax.set_ylabel("Overall Score", fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        print("No 'final_answer_reasoning' column available for additional plots.")

# =============================================================================
def plot_slope_chart_hedged_confident(df: pd.DataFrame) -> None:
    """
    Creates a slope chart that compares the mean overall scores for hedged vs. confident responses across LLMs.
    Each line represents an LLM, connecting its average overall score for hedged responses (x=0) and
    its average overall score for confident responses (x=1).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Group by LLM and response_type, computing mean overall score.
    grouped = df.groupby(["file", "response_type"])["overall_score"].mean().reset_index()
    # Pivot to have one row per LLM with separate columns for hedged and confident.
    pivot = grouped.pivot(index="file", columns="response_type", values="overall_score").reset_index()
    
    # Sort by LLM name for consistent ordering.
    pivot = pivot.sort_values("file")
    
    # Define x positions: 0 for hedged, 1 for confident.
    x_positions = [0, 1]
    
    plt.figure(figsize=(10, 6))
    
    # Plot a line for each LLM.
    for _, row in pivot.iterrows():
        llm = row["file"]
        hedged_score = row.get("hedged", np.nan)
        confident_score = row.get("confident", np.nan)
        plt.plot(x_positions, [hedged_score, confident_score], marker='o', linewidth=2, label=llm)
        # Optionally annotate the endpoints
        plt.text(x_positions[0] - 0.05, hedged_score, f"{hedged_score:.2f}", va="center", ha="right", fontsize=8)
        plt.text(x_positions[1] + 0.05, confident_score, f"{confident_score:.2f}", va="center", ha="left", fontsize=8)

    plt.xticks(x_positions, ["Hedged", "Confident"], fontsize=12)
    plt.ylabel("Mean Overall Score", fontsize=12)
    plt.title("Slope Chart: Mean Overall Score (Hedged vs. Confident) per LLM", fontsize=16)
    
    # Place legend outside the plot area.
    plt.legend(title="LLM", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()



# =============================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <path_to_csv_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Input directory '{input_dir}' does not exist or is not a directory.")
        sys.exit(1)
    
    df = load_all_files(input_dir)
    
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    plot_violin_hedged_confident(df)
    plot_stacked_bar_accept_reject(df)
    plot_violin_overall_by_file(df)
    plot_slope_chart_hedged_confident(df)
    additional_plots(df)

if __name__ == "__main__":
    main()