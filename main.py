import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.model_selection import TimeSeriesSplit
import logging


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load the sampled data
    logging.info("Loading sampled data...")

    df_ordered = pd.read_pickle("output/df_ordered.pkl")
    df_ordered = df_ordered["stock_id=4"]

    # Compute log return if df_ordered is a series of prices
    logging.info("Computing log return...")
    df_ordered = df_ordered.to_frame(name="price")
    df_ordered["log_return"] = np.log(
        df_ordered["price"] / df_ordered["price"].shift(1)
    )
    df_ordered = df_ordered.dropna(subset=["log_return"])
    logging.info("Log return computed successfully.")

    # Define the GARCH model
    def fit_garch_model(returns):
        logging.info("Fitting GARCH model...")
        model = arch_model(returns, vol="Garch", p=1, q=1)
        model_fit = model.fit(disp="off")
        logging.info("GARCH model fitted successfully.")
        return model_fit

    # Walk-forward time series cross-validation
    logging.info("Starting walk-forward time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    for fold, (train_index, test_index) in enumerate(tscv.split(df_ordered), 1):
        logging.info(f"Processing fold {fold}...")
        train, test = (
            df_ordered.iloc[train_index],
            df_ordered.iloc[test_index],
        )
        train_returns = train["log_return"]
        test_returns = test["log_return"]

        # Fit the GARCH model on the training set
        garch_model_fit = fit_garch_model(train_returns)

        # Forecast the volatility on the test set
        forecast = garch_model_fit.forecast(horizon=len(test_returns))
        predicted_volatility = forecast.variance.values[-1, :]

        # Log the predicted volatility for the test set
        # logging.info(f"Predicted volatility for fold {fold}: {predicted_volatility}")
    logging.info("Cross-validation completed.")
