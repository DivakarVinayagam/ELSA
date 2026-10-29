# ELSA
ELSA is a real-time NLP application that leverages a fine-tuned RoBERTa transformer model (roberta-base-go_emotions) to classify complex, fine-grained human emotional states from unstructured natural language text. Built for multi-label classification across 28 discrete emotional categories, the application features dynamic confidence thresholding ($\ge 15\%$) to surface primary and nuanced co-occurring emotions. ELSA delivers visual terminal output powered by rich, featuring formatted data tables and confidence meters for optimized developer and user experience.
## Key Features

* **Transformer-Based Inference:** Powered by `SamLowe/roberta-base-go_emotions` for state-of-the-art multi-label emotion prediction.
* **Fine-Grained Classification:** Predicts across 28 distinct emotional categories derived from Reddit text data.
* **Dynamic Confidence Thresholding:** Surfaces primary and secondary emotional signals using a customizable confidence filter ($\ge 15\%$).
* **Terminal UI:** Utilizes `rich` to format live predictions into styled tables and probability indicators directly in the command line.

---

## Tech Stack

* **Language:** Python 3.14+
* **Model Framework:** Hugging Face `transformers`, PyTorch (`torch`)
* **Data Processing:** `pandas`, `scikit-learn`
* **Terminal UI:** `rich`

---

## Repository Structure

```text
ELSA/
├── .gitignore              # Configured to exclude virtual environments (.venv)
├── 
├── 
├── goemotions_1.csv        # Evaluation dataset split
├── goemotions_2.csv        # Evaluation dataset split
└── goemotions_3.csv        # Evaluation dataset split
