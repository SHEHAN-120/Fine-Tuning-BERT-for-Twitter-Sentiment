# Fine-Tuning-BERT-for-Twitter-Sentiment

This repository contains code for fine-tuning **BERT (bert-base-uncased)** on a multi-class sentiment classification dataset of Twitter tweets. The model predicts different sentiment categories based on tweet text.

## 📌 Features

* Fine-tunes **BERT** for multi-class sentiment classification.
* Uses **Hugging Face Transformers** for model, tokenizer, and training pipeline.
* Implements **datasets** for data handling.
* Includes data preprocessing, tokenization, training, evaluation, and inference.
* Supports visualization with **Matplotlib** and **Seaborn** (confusion matrix, class distribution).
* Provides helper function and Hugging Face pipeline for predictions.

## 🛠️ Technologies & Frameworks Used

* **Python 3.12**
* **Hugging Face Transformers** (modeling, tokenization, training)
* **Datasets (Hugging Face)** (data handling)
* **Scikit-learn** (train/test split, metrics)
* **PyTorch** (deep learning backend)
* **Matplotlib & Seaborn** (visualizations)
* **Pandas** (data analysis)


## 🚀 Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Fine-Tuning-BERT-for-Twitter-Sentiment.git
cd Fine-Tuning-BERT-for-Twitter-Sentiment
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run training (Colab/Notebook)

The training script is in the provided notebook. It includes:

* Data loading & preprocessing
* Tokenization
* Model training with Hugging Face Trainer
* Evaluation with metrics (Accuracy, F1-score)
* Confusion matrix visualization

### 4. Run predictions

use Hugging Face pipeline:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="./bert-base-uncased-sentiment-model")
print(classifier("This was an amazing journey!"))
```

## 📊 Results

* Trained for **2 epochs** on Twitter multi-class sentiment dataset.
* Metrics include **Accuracy** and **F1-score**.
* Visualized classification report and confusion matrix.

## 💾 Model Saving

The fine-tuned model is saved using:

```python
trainer.save_model("bert-base-uncased-sentiment-model")
```

You can load it again for inference with:

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="./bert-base-uncased-sentiment-model")
```

## 📜 License

MIT License
