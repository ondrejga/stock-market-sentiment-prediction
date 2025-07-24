# Stock Market Sentiment Prediction

This project leverages state-of-the-art natural language processing (NLP) models to analyze Twitter sentiment and predict movements in popular American stocks. It processes financial tweets, extracts sentiment scores using transformer-based models, and evaluates their impact on trading strategies enabling more data-driven investment decisions.

## Objective

The primary goal is to evaluate how well four different machine learning models can predict stock price movements for major U.S. companies, specifically:

- **Apple (AAPL)**
- **Advanced Micro Devices (AMD)**
- **Amazon (AMZN)**
- **Google (GOOGL)**
- **Meta (META)**
- **Microsoft (MSFT)**
- **Procter & Gamble (PG)**
- **Tesla (TSLA)**

## Models Tested

The following four types of models were evaluated:

- **LSTM (Long Short-Term Memory)**
- **Transformer**
- **Linear Regression**
- **Tree-based Models** (XGBoost, AdaBoost)

## Features Used

The models were trained using the following stock market features:

- **Sentiment Scores**: Derived from a dataset of tweets related to the selected stocks, covering the period from 09/30/2021 to 09/30/2022 [source](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction). Sentiment for each tweet was inferred using a fine-tuned [RoBERTa model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), trained specifically to classify sentiment as positive, neutral, or negative. Daily sentiment scores were calculated as the average sentiment score of all tweets for a given stock on a given day.  
  See `sentiment_assessment.ipynb` for more details.
  
- **Technical Indicators**:
  - Simple Moving Average (SMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)

- **Lag Windows**: Number of previous days used for predicting the next day's price movement.

- **Historical Prices**: Past daily price data.

See `results_assessment.ipynb` for a deeper analysis.

## Problem Formulation

The task is modeled as a binary classification problem:  
**Given historical and derived features, predict whether a stock's price will go up or down the next day.**

### Step 1: Initial Model Comparison

A grid search was conducted across the four models with a primary focus on tuning key hyperparameters—such as the number of hidden layers for LSTM and Transformer models.

**Result**:  
Tree-based models outperformed the others in terms of both **accuracy** and **F1-score**.  
See `results_assessment.ipynb` for more details.

### Step 2: Advanced Tree-Based Model Tuning

Given their superior performance, a more extensive grid search was performed for **XGBoost** and **AdaBoost** models. For each model, I tested combinations of hyperparameters and feature subsets.

I selected the top-performing models as follows:

- 2 from AdaBoost
- 2 from XGBoost  
  (One optimized for **accuracy**, the other for a **compound score** combining accuracy and F1-score.)

These models were:

- Trained on **166 trading days**
- Tested on **76 trading days**

### Simulated Trading Strategy

To evaluate real-world effectiveness, a basic trading strategy was implemented:

- **Long Position**: If the model predicts an upward movement, buy at the beginning and sell at the end of the day.
- **Short Position**: If the model predicts a downward movement, short at the beginning and close at the end of the day.

Each model was trained and evaluated **100 times**, and the expected performance metrics were averaged.

| Model     | Optimized For     | Accuracy (%) | Expected Profit (%) |
|-----------|-------------------|--------------|----------------------|
| AdaBoost  | Accuracy           | 57.10        | 6.55                 |
| AdaBoost  | Compound Score     | 51.89        | 8.69                 |
| XGBoost   | Accuracy           | 55.81        | 6.95                 |
| XGBoost   | Compound Score     | 56.09        | **16.12**            |

### Key Insight

Surprisingly, the best-performing model (XGBoost optimized for compound score) **did not use sentiment scores or technical indicators**—only historical prices. It achieved:

- **56.09% accuracy**
- **16.12% profit** (using the basic trading strategy)

### Important Caveat

These results should be interpreted with caution. The model selection and hyperparameter tuning were done based on **test set performance**, which introduces **data leakage** and likely overestimation of real-world effectiveness.

To more accurately assess performance in a real trading environment:
- Hyperparameters should be selected using a **validation set**
- Final evaluation should be conducted on a **separate test set**

---

### Notebooks

- `sentiment_assessment.ipynb` — Sentiment extraction and analysis
- `training.ipynb` — Model training, tuning
- `results_assessment.ipynb` — Model evaluation

---

## License

MIT License

---

## Acknowledgements

- Twitter Sentiment Dataset from [Kaggle](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction)
- [Cardiff NLP RoBERTa Model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)


