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

## Project Structure

```
.
├── data
│   └── hedging_questions.csv      # CSV file with questions and responses
├── results                        # Directory where experiment results are saved
├── data_utils.py                  # Module for file I/O (loading questions, saving results)
├── evaluator.py                   # Module for LLM response evaluation functions
├── experiment.py                  # Module that orchestrates the experiment sessions
├── model_loader.py                # Module for loading models and tokenizers
├── main.py                        # Main script to run experiments across models
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```