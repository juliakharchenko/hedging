import os
import sys
import re
import pandas as pd
from typing import Optional

def normalize_csv(file_path: str, output_dir: str) -> Optional[str]:
    """
    Normalizes a CSV file by ensuring that result columns are formatted consistently.
    
    Normalizations applied:
      - For columns matching the regex pattern ^question_\\d+_score$: convert to numeric.
      - For the 'final_answer' column (if present): trim whitespace and convert to lowercase.
    
    The normalized CSV is saved in the output directory using the same file name.
    
    Args:
        file_path: Path to the input CSV file.
        output_dir: Directory to store the normalized CSV file.
    
    Returns:
        The path to the normalized CSV file if successful; otherwise, None.
    """
    df = pd.read_csv(file_path, encoding="utf-8")
    print(f"Processing {file_path}...")
    
    # Normalize score columns
    for index, row in df.iterrows():
        for col in df.columns:
            val = re.sub(r'[^a-zA-Z0-9\s]', '', str(row[col]))
            if re.match(r"^question_\d+_score$", col):
                potential_answers = ["1", "2", "3", "4", "5"] # , "unknown"
                if val.strip().lower() not in potential_answers:
                    print(f"Error: Unrecognized score value '{val}' in column '{col}'.")
                    input_str = f"Please enter a fixed value. Possible values are: {', '.join(potential_answers)}\n"
                    df.at[index, col] = int(input(input_str).strip().lower())
                else:
                    df.at[index, col] = int(val.strip().lower())
            if col == "final_answer":
                potential_answers = ["do not advance", "advance with reservations", "advance"] # , "unknown"
                if val.strip().lower() not in potential_answers:
                    print(f"Error: Unrecognized final answer value '{val}'.")
                    input_str = f"Please enter a fixed value. Possible values are: {', '.join(potential_answers)}\n"
                    df.at[index, col] = input(input_str).strip().lower()
                else:
                    df.at[index, col] = val.strip().lower()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path

def normalize_directory(input_dir: str, output_dir: str = "normalized") -> None:
    """
    Processes all CSV files in the input directory by normalizing them and saving the 
    normalized versions to the output directory.
    
    Args:
        input_dir: Directory containing CSV files.
        output_dir: Directory where normalized CSV files will be saved.
    """
    if not os.path.isdir(input_dir):
        print(f"Input directory '{input_dir}' does not exist or is not a directory.")
        sys.exit(1)
    
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        print("No CSV files found in the provided directory.")
        sys.exit(1)
    
    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        output_file = normalize_csv(file_path, output_dir)
        if output_file:
            print(f"Normalized CSV saved to: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <path_to_csv_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    normalize_directory(input_dir)

if __name__ == "__main__":
    main()
