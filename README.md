# Hedging
Hedging in LLMs
See current TODOs: https://docs.google.com/document/d/1T1-Ra-jVlcodZGsJwqRg1eXGf129TzCu_v2zqG7d_wU/edit?usp=sharing 

## Prerequisites

- **Python 3.8+**  
- **CUDA-compatible GPU** (if you plan to use GPU mode; otherwise, modify the code accordingly).
- A Hugging Face access token, which must be set as an environment variable (`HF_TOKEN`).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/juliakharchenko/hedging.git
   cd hedging
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. **Set your Hugging Face token:**  
   Set the `HF_TOKEN` environment variable with your Hugging Face access token. For example:
   ```bash
   export HF_TOKEN=your_huggingface_token_here
   ```
   On Windows, use:
   ```cmd
   set HF_TOKEN=your_huggingface_token_here
   ```

## Running the Experiment

To run the experiments on your list of models, simply execute:
```bash
python main.py
```
This will:
- Load each model specified in `main.py`
- Run the experiment sessions for each model
- Save the results into separate subfolders under the `results` directory

Below is an example section you can add to your `README.md` to explain how to use the `analysis.py` script:

---

## Analyzing Experiment Results

The `analysis.py` script is designed to analyze the results from the above interview experiments.

### Usage

To run the analysis, execute the script from the command line and provide the path to your CSV results file as an argument:

```bash
python analysis.py path/to/your/interview_scores_modelname.csv
```

The script will perform the following steps:
1. **Data Loading:**  
   It reads the CSV file containing interview data. The CSV should include columns such as `interview_round`, `hedged_or_confident`, and various question score columns (e.g., `question_1_score`).

2. **Data Preparation:**  
   It converts the score columns to numeric values and computes an average score for each interview session.

3. **Statistical Analysis:**  
   It separates the data into "Hedged" and "Confident" groups, performs an independent t‑test to compare the average scores between these groups, and prints the t-statistic and p‑value.

4. **Data Visualization:**  
   - **Boxplot:** Generates and saves a boxplot (`avg_score_boxplot.png`) that visualizes the distribution of average scores for each response type.
   - **Stacked Bar Chart:** (If available) Creates a stacked bar chart (`final_decision_stacked_bar.png`) showing the distribution of final decisions across response types.

5. **Output:**  
   The script prints statistical results to the console and displays the generated plots. The visualization images are also saved in the same directory where the script is executed.

---

## Project Structure

```
.
├── data
│   └── hedging_questions.csv      # CSV file with questions and responses
├── results                        # Directory where experiment results are saved
│   ├── ...
│   └── interview_scores_{model}.csv  # Interview scores for model {model}
├── code
│   ├── data_utils.py                  # Module for file I/O (loading questions, saving results)
│   ├── evaluator.py                   # Module for LLM response evaluation functions
│   ├── experiment.py                  # Module that orchestrates the experiment sessions
│   ├── model_loader.py                # Module for loading models and tokenizers
│   ├── main.py                        # Main script to run experiments across models
│   └── analysis.py                    # Script to analyze interview results
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```