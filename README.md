# HC3 Detector

This project trains a machine learning model to distinguish **human-written** and **AI-generated** answers.  
The dataset is the [HC3 corpus](https://huggingface.co/datasets/Hello-SimpleAI/HC3), which contains questions paired with both human answers (from Reddit, WikiQA, etc.) and ChatGPT answers.

---

## How it works

- Text is cleaned and deduplicated.
- Two types of features are created:
  - **TF-IDF vectors** (unigrams + bigrams).
  - **Stylistic features** (average sentence length, type-token ratio, pronoun rate).
- A **Logistic Regression** model combines these features.
- The model is trained, evaluated, and saved for reuse.

---

## Project structure

``` bash
hc3-detector/
├── src/
│ ├── get_data.py # download and save dataset
│ ├── features.py # feature engineering
│ ├── baseline.py # training script
│ └── predict.py # run predictions on new text
├── notebooks/ # exploratory notebooks
├── data/ # dataset (ignored in git)
├── models/ # trained model
├── results/ # metrics and plots
├── requirements.txt # dependencies
└── README.md
```

---

## Installation

Set up a Python virtual environment and install requirements:

```bash
pip install -r requirements.txt
```

If you want the exact versions used in development, use:

```bash
pip install -r requirements_pinned.txt
```

---

## Training

To train the model from scratch:

```bash
python src/baseline.py --data data/hc3_full.parquet
```

This will save:
- Metrics to results/metrics.json
- Confusion matrix plot
- Feature importances
- Trained model to models/hc3_lr.joblib

---

## Prediction

Run the model on a custom piece of text:

```bash
python src/predict.py "Honestly I wasn’t sure what to expect when I started..."
```

To see the top features behind the prediction:

```bash
python src/predict.py "It is important to note that reproducibility..." --explain
```

---

## Results

- Test accuracy on HC3: ~96%
- ROC-AUC (AI Positive Class): ~0.94

---

## Limitations

- The dataset uses early ChatGPT outputs (2022).
- Some modern AI text looks more human-like and may not be flagged.
- Vocab richness strongly affects the decision boundary.

## Future Work

I may add more stylistic features (e.g. readability) and test on more recent LLM outputs.

