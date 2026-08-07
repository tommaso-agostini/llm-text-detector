# Italian Text Classification — Project Overview

This repository contains a series of experiments for **binary text classification on Italian text**, based on the **DeSeGMa-IT shared task (subtask A)** dataset. Four different modeling approaches are compared: classic n-gram based SVMs, linguistically-informed features, static word embeddings, and fine-tuned Transformer language models. Each approach is implemented and evaluated in its own notebook, on top of a shared data-preparation pipeline.

## Project Structure

```
.
├── data/
│   ├── desegma-it.subTaskA.shared.train.0923-1220.csv
│   ├── desegma-it.subTaskA.shared.test.1117_1835.csv
│   ├── desegma-it.subTaskA.with_labels.test.1117_1835.csv
│   ├── processed/
│   │   ├── df_train_processed.pkl
│   │   ├── df_val_processed.pkl
│   │   ├── df_test_processed.pkl
│   │   └── data_task2/
│   │       ├── all_data_profiling/          # per-document .txt files + .zip archive
│   │       └── profiling_UD_all_conllu/     # CoNLL-U output from Profiling-UD
│   └── word_embeddings/
│       ├── itwac128.sqlite
│       └── itwac128.txt
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── task_1.ipynb
│   ├── task_2.ipynb
│   ├── task_3.ipynb
│   └── task_4.ipynb
├── classes.py                 # Token / Sentence / Document data model
└── finetuned_model/           # output of task_4 (Transformer checkpoints)
```

> Note: paths in the notebooks are relative (`../data/...`), so each notebook is expected to be run from a `notebooks/` subfolder with a sibling `data/` directory as shown above.

## Data Preparation (`data_preparation.ipynb` + `classes.py`)

This notebook builds the shared, pre-processed dataset used by all four modeling notebooks, starting from the raw DeSeGMa-IT subtask A CSV files:

1. **Balanced split creation.** From the original training CSV, 1000 documents per class (label `0`/`1`) are sampled to build a balanced training set; a further 500+500 documents are sampled from the remaining training data (with no overlap) to build the validation set. The test set is built the same way (500+500) from the separate, labeled test CSV.
2. **Export for Profiling-UD.** Every document (train/val/test) is written to its own `.txt` file, named with its split and original dataframe index (e.g. `train_15.txt`), and all files are packaged into a single `.zip` archive. This naming convention allows exact re-alignment of documents to their labels later on, avoiding any data leakage between splits. The archive must be uploaded manually to the [Profiling-UD](http://linguistic-profiling.italianlp.it/) web service, which returns a folder of per-document **CoNLL-U** annotation files (word, lemma, POS, and morpho-syntactic features).
3. **CoNLL-U parsing.** The returned CoNLL-U files are parsed using the `Document` / `Sentence` / `Token` classes defined in `classes.py` (see below), which reconstruct each document's tokens, lemmas, POS tags, and original raw text.
4. **Re-alignment and export.** Parsed documents are split back into train/val/test using the encoded index, the correct label is reattached, and string-joined versions of tokens/lemmas/POS tags are added as the `tokens_processed`, `lemmas_processed`, and `pos_processed` columns (alongside the original `text` and `label` columns). The three resulting dataframes are saved as `df_train_processed.pkl`, `df_val_processed.pkl`, and `df_test_processed.pkl` — the exact files loaded by tasks 1–4.

### `classes.py` — data model

Defines the object model used to represent CoNLL-U annotated documents:

- **`Token`** — a single annotated token, storing its surface form (`word`), `lemma`, and `pos` tag; exposes `num_chars()`.
- **`Sentence`** — an ordered collection of `Token` objects plus the original sentence `text`; exposes helpers to retrieve all words/lemmas/POS tags (`get_words`, `get_lemmas`, `get_pos_tags`) and aggregate counts (`num_tokens`, `num_chars`).
- **`Document`** — a full document (`path`, `doc_id`, `split`, `label`), composed of `Sentence` objects. `load_sentences_from_conllu()` parses a `.conllu` file into `Sentence`/`Token` objects, skipping multiword tokens and comment lines. Convenience methods aggregate words, lemmas, POS tags, token/sentence/character counts across the whole document.

All four modeling notebooks load the same pre-processed dataset (`df_train_processed.pkl`, `df_val_processed.pkl`, `df_test_processed.pkl`) produced by this pipeline, containing the `text`, `tokens_processed`, `lemmas_processed`, `pos_processed`, and `label` columns.

## Tasks

### Task 1 — Linear SVM with n-grams (`task_1.ipynb`)
Implements a **LinearSVC** classifier on top of **TF-IDF** features, comparing several granularities and n-gram ranges:
- Word forms (tokens): unigrams and bigrams
- Lemmas: unigrams and bigrams
- Part-of-Speech tags: bigrams and trigrams
- Characters: 3–5 gram sequences

Each configuration is evaluated via 5-fold cross-validation (macro-F1) on the training set. The best-performing configuration (character 3–5-grams) is then retrained with different regularization strengths (`C`) and evaluated on the train/validation/test splits, including classification reports and a confusion matrix.

### Task 2 — Linear SVM on non-lexical linguistic features (`task_2.ipynb`)
Uses hundreds of **non-lexical linguistic features** extracted with **Profiling-UD** (e.g. syntactic and morphological statistics) instead of lexical content. Since features have very different scales and units, a **StandardScaler** is applied before feeding a **LinearSVC** classifier. The notebook aligns the Profiling-UD CSV output with the original train/validation/test splits (via document IDs), trains a baseline model, and reports performance (classification report + confusion matrices) on training, validation, and test sets.

### Task 3 — Linear SVM with word embeddings (`task_3.ipynb`)
Builds document representations from **pre-trained static word embeddings** (128-dimensional, trained on the *itWaC* Italian corpus), stored in a SQLite database and exported to a text file for faster loading. Documents are vectorized by pooling (mean, sum, or max) the embeddings of tokens filtered by specific POS tags (e.g. NOUN/ADJ/VERB or VERB/NOUN/INTJ/DET). A normalization step handles URLs, numbers, and unusually long tokens before lookup. The resulting document vectors are scaled and classified with a **LinearSVC**, with performance measured across pooling strategies and POS configurations.

### Task 4 — Fine-tuning a Transformer language model (`task_4.ipynb`)
Fine-tunes **`dbmdz/bert-base-italian-cased`** (an Italian BERT model) for binary sequence classification using the Hugging Face `transformers`, `datasets`, and `evaluate` libraries. Key steps:
- Deterministic setup (fixed seeds for Python, NumPy, and PyTorch)
- Tokenization with padding/truncation to a maximum length of 512 tokens
- Training with the Hugging Face `Trainer` API, evaluating Accuracy and F1 at the end of each epoch, for 3 epochs
- Evaluation and confusion matrix on the test set

GPU availability is checked at the start of the notebook; a CUDA-enabled GPU is strongly recommended for this task.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- `torch`, `transformers`, `datasets`, `accelerate`, `evaluate` (for Task 4)
- `sqlite3` (standard library, used in Task 3)

Install the main dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install torch transformers datasets accelerate evaluate
```

## Running the Notebooks

1. **Data preparation (run first):**
   - Place the three raw DeSeGMa-IT subtask A CSV files under `data/`.
   - Run `data_preparation.ipynb` (with `classes.py` importable from the same folder) to build the balanced train/val/test splits.
   - Upload the generated `all_data_profiling.zip` to [Profiling-UD](http://linguistic-profiling.italianlp.it/), download the resulting CoNLL-U folder, and place it at `data/processed/data_task2/profiling_UD_all_conllu/` (this manual step cannot be automated).
   - Finish running the notebook to parse the CoNLL-U files and produce `df_train_processed.pkl`, `df_val_processed.pkl`, and `df_test_processed.pkl` under `data/processed/`.
2. For Task 3, make sure `data/word_embeddings/itwac128.sqlite` is available (the `.txt` export is generated automatically on first run if missing).
3. Open and run each modeling notebook independently from the `notebooks/` folder:
   - `task_1.ipynb` — n-gram based SVM
   - `task_2.ipynb` — linguistic-feature based SVM
   - `task_3.ipynb` — word-embedding based SVM
   - `task_4.ipynb` — fine-tuned Italian BERT
4. Task 4 will save the fine-tuned model checkpoints under `../finetuned_model`.

## Evaluation Metrics

All notebooks report standard classification metrics — precision, recall, F1-score (including macro-F1) — via `classification_report`, together with confusion matrices on the validation and/or test sets, allowing direct comparison of the four modeling approaches on the same data splits.

## Conclusions & Key Takeaways

This project evaluated four distinct approaches for machine-generated text detection in Italian, highlighting the trade-offs between traditional feature engineering and modern deep learning. All tested models successfully identified linguistic signals to discriminate between human and AI-generated text, significantly outperforming a random baseline. 

**Key Findings:**
* **Deep Learning Generalizes Best (Task 4):** The fine-tuned Italian BERT model was the most powerful and effective solution, achieving the highest test set performance (Macro F1 = 0.920). The Transformer's self-attention mechanism captured complex contextual dependencies autonomously, proving superior generalization capabilities without the need for manual feature engineering.
* **Lexical Overfitting (Task 1):** While character n-grams achieved the highest validation score (Macro F1 = 0.987), they experienced the sharpest performance drop on the test set (Macro F1 = 0.850). This suggests that while n-grams excel at finding local orthographic patterns, they are brittle and tend to overfit to the specific vocabulary of the training domain.
* **Robustness of Abstract Features (Tasks 2 & 3):** Models utilizing non-lexical syntactic features (Profiling-UD) and global semantic representations (static word embeddings) demonstrated strong robustness to domain shifts. Both maintained a stable Macro F1 of 0.880 on the test set, confirming that focusing on the abstract "structure" of the text helps limit dependency on specific tokens.

Overall, while traditional linguistic representations yield effective and highly interpretable models, incorporating pre-trained contextual information (Transformers) provides the best adaptability to unseen data.
