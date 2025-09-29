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
Tree-based models without sentiment and technical indicators features outperformed the others in terms of both **accuracy** and **F1-score**.  
See `results_assessment.ipynb` for more details.

### Step 2: Advanced Tree-Based Model Backtesting

Since tree-based models showed the most promise, I backtested XGBoost and AdaBoost using the following procedure:

## Backtesting procedure

- **Period:** Weekly re-training and evaluation from **2022-12-10** to **2024-08-24**.  
- **Data window:** For each stock, use the **past three years** of price data.  
- **Validation split:** The **most recent 30 days** are held out as a validation set; the remaining history is used for training.  
- **Sample construction:** Each data point is a sliding-window sample with lagged features (e.g., price_{t-4}, price_{t-3}, price_{t-2}, price_{t-1}, price_t).  
  - **Features:** volume, close, open, high, low (for the lagged days).  
  - **Target:** binary label indicating whether the next-day price (t+1) is higher than the current-day price (t).

## Model selection & deployment

- For each week and each stock, perform hyperparameter tuning for XGBoost and AdaBoost on the training + validation split.  
- Select the model and hyperparameters that maximize **profit** on the validation set.  
- “Deploy” the selected model for one week to generate predictions.  
- Maintain one separate model per stock.

Across the full backtesting period this procedure produced **720 predictions**.

## Results

- **Simulated trading rule:** go long when the model predicts an upward move; go short otherwise.  
- **Average weekly return (simulated):** **0.0547%**  
- **Mean prediction accuracy:** **49.79%**

With only simple price and volume features, the strategy produced results consistent with random guessing and did not generate a reliable edge.

## Recommendations / Next steps

- **Add richer features:** incorporate technical indicators, Twitter sentiment, financial news, macroeconomic variables, and alternative data sources.  
- **Backtest other architectures:** run equivalent backtests for LSTM and Transformer models — although they underperformed on static tests, expanded features or temporal modeling may improve their performance.  
- **Try alternative prediction targets:** consider multi-day horizons, regression on price change magnitude, or probabilistic outputs instead of binary labels.  
- **Refine evaluation metric:** optimize directly for trading performance (e.g., risk-adjusted returns) rather than only accuracy.

See `backtesting.ipynb` for full implementation details and results.


---

### Notebooks

- `sentiment_assessment.ipynb` — Sentiment extraction and analysis
- `training.ipynb` — Model training, tuning
- `results_assessment.ipynb` — Model evaluation
- `backtesting.ipynb`  — Model backtesting

---

## License

MIT License

---

## Acknowledgements

- Twitter Sentiment Dataset from [Kaggle](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction)
- [Cardiff NLP RoBERTa Model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)


