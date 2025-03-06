import numpy as np
import pandas as pd


def comprehensive_feature_engineering(
    df_order: pd.DataFrame, df_trade: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate a comprehensive set of features for volatility prediction based on order book and trade data.
    Aggregates features at the 150-second, 300-second, and 450-second levels.
    """
    # Order Table Features

    # Price Velocity and Acceleration (Speed Features)
    df_order["wap1"] = (
        df_order["bid_price1"] * df_order["ask_size1"]
        + df_order["ask_price1"] * df_order["bid_size1"]
    ) / (df_order["bid_size1"] + df_order["ask_size1"])
    df_order["log_return1"] = (
        df_order.groupby("time_id")["wap1"]
        .apply(lambda x: np.log(x / x.shift(1)))
        .values
    )
    df_order["price_velocity"] = df_order["wap1"].diff()
    df_order["price_acceleration"] = df_order["price_velocity"].diff()

    # Pressure and Imbalance Features
    df_order["volume_imbalance"] = abs(
        (df_order["ask_size1"] + df_order["ask_size2"])
        - (df_order["bid_size1"] + df_order["bid_size2"])
    )
    df_order["order_book_imbalance"] = df_order["bid_size1"] / (
        df_order["bid_size1"] + df_order["ask_size1"]
    )
    df_order["cumulative_depth_imbalance"] = (
        df_order["bid_size1"] + df_order["bid_size2"]
    ) / (
        (df_order["bid_size1"] + df_order["bid_size2"])
        + (df_order["ask_size1"] + df_order["ask_size2"])
    )

    # Liquidity Features
    df_order["bid_ask_spread"] = df_order["ask_price1"] - df_order["bid_price1"]
    df_order["price_spread"] = (df_order["ask_price1"] - df_order["bid_price1"]) / (
        (df_order["ask_price1"] + df_order["bid_price1"]) / 2
    )
    df_order["order_book_slope"] = (
        df_order["bid_price2"] - df_order["bid_price1"]
    ) / df_order["bid_size1"]

    # Realized Volatility
    df_order["realized_volatility"] = df_order.groupby("time_id")[
        "log_return1"
    ].transform(np.std)

    # Define aggregation functions
    order_features = {
        "wap1": [np.sum, np.mean, np.std],
        "log_return1": [np.sum, np.mean, np.std],
        "price_spread": [np.sum, np.mean, np.std],
        "volume_imbalance": [np.sum, np.mean, np.std],
        "bid_ask_spread": [np.sum, np.mean, np.std],
        "price_velocity": [np.mean, np.std],
        "price_acceleration": [np.mean, np.std],
        "order_book_imbalance": [np.mean],
        "cumulative_depth_imbalance": [np.mean],
        "order_book_slope": [np.mean],
        "realized_volatility": [np.mean],
    }

    # Trade Table Features

    # Price Velocity and Acceleration (Speed Features)
    df_trade["log_return"] = (
        df_trade.groupby("time_id")["price"]
        .apply(lambda x: np.log(x / x.shift(1)))
        .values
    )

    # Realized Volatility
    df_trade["realized_volatility"] = df_trade.groupby("time_id")[
        "log_return"
    ].transform(np.std)

    # Trade Execution and Impact Features
    df_trade["trade_volume"] = df_trade["size"]
    df_trade["trade_impact_score"] = df_trade["size"] / (
        df_order["bid_size1"] + df_order["ask_size1"]
    )
    df_trade["trade_intensity"] = df_trade.groupby("time_id")["size"].transform("count")

    # Define aggregation functions for trade features
    trade_features = {
        "log_return": [np.sum, np.mean, np.std],
        "trade_volume": [np.sum, np.mean, np.std],
        "trade_impact_score": [np.mean],
        "trade_intensity": [np.mean],
        "realized_volatility": [np.mean],
    }

    # Timeframes for aggregation
    timeframes = [150, 300, 450]
    df_combined = []

    for timeframe in timeframes:
        # Aggregate Order Features
        order_agg = (
            df_order[df_order["seconds_in_bucket"] >= timeframe]
            .groupby("time_id")
            .agg(order_features)
            .reset_index()
        )

        # Aggregate Trade Features
        trade_agg = (
            df_trade[df_trade["seconds_in_bucket"] >= timeframe]
            .groupby("time_id")
            .agg(trade_features)
            .reset_index()
        )

        # Merge Order and Trade Features for the current timeframe
        df_agg = pd.merge(
            order_agg,
            trade_agg,
            on="time_id",
            how="left",
            suffixes=(f"_order_{timeframe}", f"_trade_{timeframe}"),
        )
        df_combined.append(df_agg)

    # Concatenate all timeframes into a single DataFrame
    df_final = pd.concat(df_combined, axis=1)

    return df_final


# Example usage
# order_df = pd.read_parquet('path_to_order_data.parquet')
# trade_df = pd.read_parquet('path_to_trade_data.parquet')
# df = comprehensive_feature_engineering(order_df, trade_df)
