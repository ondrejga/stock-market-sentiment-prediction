import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date, time
from dateutil.relativedelta import relativedelta

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import ParameterGrid
import xgboost as xgb
import os


class StockPredictor:
    def __init__(self, tickers, lag=5, years=5, log_file="predictions_log.csv"):
        self.tickers = tickers
        self.lag = lag
        self.years = years
        self.models = {}        # {ticker: {model_name: model}}
        self.predictions = {}   # {ticker: {model_name: (probs, pred)}}
        self.data = {}
        self.log_file = log_file

        # Default hyperparams
        self.hyperparams = {
            "XGBoost": {
                'max_depth': [3],
                'min_child_weight': [1],
                'gamma': [0.2],
                'learning_rate': [0.10],
                'n_estimators': [100]
            },
            "AdaBoost": {
                'estimator__max_depth': [5],
                'learning_rate': [1],
                'n_estimators': [50]
            }
        }

        # If log file doesn't exist, create with headers
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=["Date", "Ticker", "Model", "Prediction", "Up_Prob", "Down_Prob"]).to_csv(
                self.log_file, index=False
            )

    def load_and_process_stock_data(self, ticker, start_date, end_date):
        """Download stock data and clean."""
        delta = timedelta(days=1)
        stock_data = yf.download(ticker, start=start_date, end=end_date + delta)
        stock_data.columns = stock_data.columns.droplevel(level=1)  # Remove multi-index
        stock_data.reset_index(inplace=True)
        stock_data['Date'] = stock_data['Date'].dt.date
        return stock_data

    def make_sequences(self, data):
        """Create lag sequences."""
        X, y, to_pred = [], [], []
        features = ['Close', 'High', 'Low', 'Open', 'Volume']

        for idx in range(len(data) - self.lag):
            if idx + self.lag + 1 >= len(data):
                to_pred = data.iloc[idx:idx + self.lag][features].values.flatten()
                continue

            X.append(data.iloc[idx:idx + self.lag][features].values)
            label = 1 if data.iloc[idx + self.lag - 1]['Close'] < data.iloc[idx + self.lag]['Close'] else 0
            y.append(label)

        return np.array(X), np.array(y), np.array([to_pred])

    def create_model(self, model_name, params=None):
        """Factory for models."""
        if model_name == "XGBoost":
            return xgb.XGBClassifier(
                use_label_encoder=False, eval_metric="logloss", verbosity=0, **(params or {})
            )
        elif model_name == "AdaBoost":
            base_estimator = DecisionTreeClassifier(
                max_depth=params.get("max_depth", 1) if params else 1
            )
            return AdaBoostClassifier(
                estimator=base_estimator,
                learning_rate=params.get("learning_rate", 1.0),
                n_estimators=params.get("n_estimators", 50),
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def train_and_predict(self, model_name, model, X_train, y_train, to_pred):
        """Fit model and return predictions."""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        model.fit(X_train_flat, y_train)

        to_pred = to_pred.reshape(1, -1)
        y_pred_probs = model.predict_proba(to_pred)
        y_pred = model.predict(to_pred)
        return y_pred_probs, y_pred

    def daily_train(self):
        """Retrain all tickers with all models and make predictions for tomorrow."""
        end_date = date.today()
        start_date = end_date - relativedelta(years=self.years)

        start_datetime = datetime.combine(start_date, time.min)
        end_datetime = datetime.combine(end_date, time.min)

        for ticker in self.tickers:
            # Load data
            df = self.load_and_process_stock_data(ticker, start_datetime, end_datetime)
            self.data[ticker] = df

            # Make sequences
            X_train, y_train, to_pred = self.make_sequences(df)

            self.models[ticker] = {}
            self.predictions[ticker] = {}

            # Train both models
            for model_name in ["XGBoost", "AdaBoost"]:
                best_model, best_score, best_pred = None, -np.inf, None
                for params in ParameterGrid(self.hyperparams[model_name]):
                    model = self.create_model(model_name, params=params)
                    y_pred_probs, y_pred = self.train_and_predict(model_name, model, X_train, y_train, to_pred)
                    score = y_pred_probs[0][1]  # probability of upward movement

                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_pred = (y_pred_probs, y_pred)

                self.models[ticker][model_name] = best_model
                self.predictions[ticker][model_name] = best_pred

                # Save prediction to CSV
                self._log_prediction(ticker, model_name, best_pred)

    def _log_prediction(self, ticker, model_name, prediction):
        """Append prediction to CSV log."""
        probs, pred = prediction
        new_row = pd.DataFrame([{
            "Date": date.today(),
            "Ticker": ticker,
            "Model": model_name,
            "Prediction": int(pred[0]),
            "Up_Prob": float(probs[0][1]),
            "Down_Prob": float(probs[0][0])
        }])
        new_row.to_csv(self.log_file, mode='a', header=False, index=False)

    def get_predictions(self):
        """Return predictions for all tickers + models."""
        return self.predictions


# ------------------ Usage ------------------
if __name__ == "__main__":
    tickers = ['TSLA', 'MSFT', 'PG', 'META', 'AMZN', 'GOOG', 'AMD', 'AAPL']

    predictor = StockPredictor(tickers, log_file="predictions_log.csv")
    predictor.daily_train()
    print("✅ Daily predictions logged to predictions_log.csv")
    
