# Customer Feedback Sentiment Analyzer

A machine learning project that classifies customer feedback as **Positive**, **Neutral**, or **Negative**, and flags common complaint categories such as flight delays, baggage issues, and customer service problems. It ships with an interactive Gradio web app for trying the model on your own text.

Built and trained on the [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) dataset (~14,600 tweets about major US airlines).

## Features

- **Sentiment classification** using TF-IDF features and Logistic Regression
- **Confidence score** for every prediction
- **Complaint category detection** (keyword-based): Flight Delay, Baggage, Customer Service, Cancellation, Booking
- **Interactive Gradio UI** with example inputs and a shareable public link
- **Evaluation tooling**: accuracy, classification report, and confusion matrix

## Dataset

| Sentiment | Tweets |
|-----------|--------|
| Negative  | 9,178  |
| Neutral   | 3,099  |
| Positive  | 2,363  |

The dataset is downloaded automatically through `kagglehub`; no manual download is needed. Note the strong class imbalance toward negative feedback.

## How It Works

1. **Load data**: only the `text` and `airline_sentiment` columns are used.
2. **Clean text**: lowercase, remove URLs, @mentions, hashtag symbols, numbers, and special characters, and normalize whitespace.
3. **Split**: 80/20 stratified train/test split (11,712 train / 2,928 test).
4. **Vectorize**: TF-IDF with up to 5,000 features, English stop words removed, unigrams and bigrams.
5. **Train**: Logistic Regression (`max_iter=1000`).
6. **Analyze**: predict sentiment with a confidence score, then run keyword matching to tag complaint categories.
7. **Serve**: wrap the pipeline in a Gradio interface.

## Results

Overall accuracy: **76.95%**

| Class    | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.79      | 0.94   | 0.86     | 1835    |
| Neutral  | 0.63      | 0.42   | 0.51     | 620     |
| Positive | 0.82      | 0.56   | 0.66     | 473     |

The model is strongest on negative feedback and weaker on neutral and positive, largely because of class imbalance.

## Getting Started

The notebook was developed in Google Colab, but it also runs locally.

### Requirements

- Python 3.8+
- pandas, numpy, matplotlib
- scikit-learn
- gradio
- kagglehub

### Installation

```bash
pip install pandas numpy matplotlib scikit-learn gradio kagglehub
```

### Run

1. Open `customerfeedback.ipynb` in Google Colab or Jupyter.
2. Run all cells in order. The dataset downloads automatically.
3. The last cell launches the Gradio app and prints a local URL (and a temporary public URL when `share=True`).

> The Kaggle dataset download may require Kaggle credentials depending on your environment.

## Usage

Call the analysis function directly:

```python
analyse_feedback("My flight was delayed for 4 hours and my baggage was also lost.")
```

Output:

```
Sentiment: NEGATIVE 😞

Confidence: 99.52%

Complaint Category: Flight Delay, Baggage
```

Or use the Gradio web interface: paste feedback into the text box and view the sentiment, confidence, and complaint categories.

## Project Structure

```
.
├── customerfeedback.ipynb   # Data prep, training, evaluation, and Gradio app
└── README.md
```

## Limitations and Future Work

- Complaint detection is keyword-based and can miss paraphrased complaints or produce false matches (for example, "late" inside other words).
- Neutral and positive recall is low; consider class weighting (`class_weight="balanced"`), resampling, or threshold tuning.
- Try stronger models such as linear SVM, gradient boosting, or fine-tuned transformers (e.g. DistilBERT).
- Replace keyword complaint tagging with a model trained on the dataset's `negativereason` labels.
- Preserve negation and punctuation cues (for example "not good", "!!!") in preprocessing.
- Deploy permanently on Hugging Face Spaces using `gradio deploy`.

## Acknowledgements

- Dataset: [CrowdFlower Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- Built with scikit-learn and Gradio
