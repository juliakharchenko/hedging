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
        # inner="quartile"
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
    sorted_llms = df.groupby("file")["overall_score"].median().sort_values().index
    # plt.figure(figsize=(10, 6))
    # ax = sns.violinplot(
    #     x="file",
    #     y="overall_score",
    #     data=df,
    #     order=sorted_llms,
    #     palette="coolwarm",
    #     inner="quartile"
    # )
    # do the same as above but add an extra separation between response_type
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        x="file",
        y="overall_score",
        data=df,
        order=sorted_llms,
        hue="response_type",
        palette="Set1",
        split=True,
        inner="quartile"
    )
    ax.set_title("Overall Score Distribution per LLM", fontsize=16)
    ax.set_xlabel("LLM", fontsize=14)
    ax.set_ylabel("Overall Score", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# =============================================================================
def score_vs_reasoning_length(df: pd.DataFrame) -> None:
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
def plot_violin_score_by_question_contents(df: pd.DataFrame) -> None:
    """
    Creates two horizontal plots showing the distribution of scores (1-5) for each question.
    
    The function first extracts each question's text and corresponding score (from columns named like
    "question_X" and "question_X_score"), then filters out questions with fewer than 5 responses.
    
    It computes the mean score for each question and uses that to select the top 10 easiest 
    (highest mean score) and top 10 hardest (lowest mean score) questions. 
    Both plots share the same x-axis so that distributions can be compared directly.
    """
    # Find all columns that match the pattern for question scores.
    score_cols = [col for col in df.columns if re.match(r'^question_\d+_score$', col)]
    if not score_cols:
        print("No question score columns found in the DataFrame.")
        return

    # Build a long DataFrame with columns: "question" and "score"
    data_list = []
    for score_col in score_cols:
        # Assume the corresponding question text is in the column without the "_score" suffix.
        question_col = score_col.replace("_score", "")
        if question_col in df.columns:
            data = df[[question_col, score_col]].copy()
            data.columns = ["question", "score"]
            data_list.append(data)
    if not data_list:
        print("No question data extracted.")
        return

    # Concatenate data for all questions.
    data_long = pd.concat(data_list, ignore_index=True)
    # Filter out questions with fewer than 5 responses.
    data_long = data_long[data_long.groupby('question')['score'].transform('count') > 5]

    # Compute the mean score per question.
    mean_scores = data_long.groupby("question")["score"].mean().reset_index()

    # Select easiest (highest mean scores) and hardest (lowest mean scores) questions.
    easiest_questions = mean_scores.sort_values("score", ascending=False).head(10)["question"]
    hardest_questions = mean_scores.sort_values("score", ascending=True).head(10)["question"]

    easiest_data = data_long[data_long["question"].isin(easiest_questions)]
    hardest_data = data_long[data_long["question"].isin(hardest_questions)]

    # Define a common x-axis range since scores are between 1 and 5.
    x_limits = (0.8, 5.2)

    # Create two subplots (vertical layout) with shared x-axis.
    fig, axs = plt.subplots(2, 1, figsize=(12, 16), sharex=True)

    # Plot easiest questions.
    sns.boxplot(x="score", y="question", data=easiest_data, color="lightgray",
                orient="h", whis=[0, 100], ax=axs[0])
    sns.stripplot(x="score", y="question", data=easiest_data, size=4,
                  color="black", jitter=True, orient="h", ax=axs[0])
    axs[0].set_title("Top 10 Easiest Questions by Mean Score", fontsize=16)
    axs[0].set_xlabel("")  # Remove x-label on the top plot.
    axs[0].set_ylabel("Question", fontsize=14)
    axs[0].set_xlim(x_limits)

    # Plot hardest questions.
    sns.boxplot(x="score", y="question", data=hardest_data, color="lightgray",
                orient="h", whis=[0, 100], ax=axs[1])
    sns.stripplot(x="score", y="question", data=hardest_data, size=4,
                  color="black", jitter=True, orient="h", ax=axs[1])
    axs[1].set_title("Top 10 Hardest Questions by Mean Score", fontsize=16)
    axs[1].set_xlabel("Score (1-5)", fontsize=14)
    axs[1].set_ylabel("Question", fontsize=14)
    axs[1].set_xlim(x_limits)

    plt.tight_layout()
    plt.show()

# =============================================================================
# Plot average interview score vs. final decision
def plot_avg_score_vs_decision(df: pd.DataFrame) -> None:
    """
    Creates a bar plot showing the average interview score for each final decision.
    """
    avg_scores = df.groupby("final_answer")["overall_score"].mean().sort_values()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=avg_scores.values, y=avg_scores.index, palette="coolwarm")
    ax.set_title("Average Interview Score by Final Decision", fontsize=16)
    
    ax.set_xlabel("Average Score", fontsize=14)
    plt.tight_layout()
    plt.show()

# =============================================================================
# Make a plot where I can see per question the spread of scores (maybe per resposne too)
def plot_violin_score_by_question(df: pd.DataFrame) -> None:
    """
    Creates a violin plot comparing the distribution of scores for each question.
    """
    # Find all columns that match the pattern for question scores.
    score_cols = [col for col in df.columns if re.match(r'^question_\d+_score$', col)]
    if not score_cols:
        print("No question score columns found in the DataFrame.")
        return

    # Build a long DataFrame with columns: "question" and "score"
    data_list = []
    for score_col in score_cols:
        # Assume the corresponding question text is in the column without the "_score" suffix.
        question_col = score_col.replace("_score", "")
        if question_col in df.columns:
            data = df[[question_col, score_col]].copy()
            data.columns = ["question", "score"]
            data_list.append(data)
    if not data_list:
        print("No question data extracted.")
        return

    # Concatenate data for all questions.
    data_long = pd.concat(data_list, ignore_index=True)
    # Filter out questions with fewer than 5 responses.
    data_long = data_long[data_long.groupby('question')['score'].transform('count') > 5]

    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        x="score",
        y="question",
        data=data_long,
        palette="coolwarm",
        inner="quartile"
    )
    ax.set_title("Score Distribution per Question", fontsize=16)
    ax.set_xlabel("Score", fontsize=14)
    ax.set_ylabel("Question", fontsize=14)
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
    score_vs_reasoning_length(df)
    plot_violin_score_by_question_contents(df)
    plot_avg_score_vs_decision(df)
    plot_violin_score_by_question(df)

if __name__ == "__main__":
    main()