import numpy as np
import pandas as pd


def comprehensive_feature_engineering(
    df_order: pd.DataFrame, df_trade: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate a comprehensive set of features for volatility prediction based on order book and trade data.
    Now includes dollar‐weighted averages for several trade metrics (log_return, trade_volume, trade_impact_score,
    realized_volatility).
    """

    # -------------------------- #
    # 1. Order‐Book Features    #
    # -------------------------- #

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

    # -------------------------- #
    # 2. Trade‐Data Features    #
    # -------------------------- #

    df_trade["log_return"] = (
        df_trade.groupby("time_id")["price"]
        .apply(lambda x: np.log(x / x.shift(1)))
        .values
    )
    df_trade["realized_volatility"] = df_trade.groupby("time_id")[
        "log_return"
    ].transform(np.std)

    df_trade["trade_volume"] = df_trade["size"]
    df_trade["trade_impact_score"] = df_trade["size"] / (
        df_order["bid_size1"] + df_order["ask_size1"]
    )
    df_trade["trade_intensity"] = df_trade.groupby("time_id")["size"].transform("count")

    trade_features = {
        "log_return": [np.sum, np.mean, np.std],
        "trade_volume": [np.sum, np.mean, np.std],
        "trade_impact_score": [np.mean],
        "trade_intensity": [np.mean],
        "realized_volatility": [np.mean],
    }

    # ------------------------------- #
    # 2A. Dollar‐Weighted Averages   #
    # ------------------------------- #
    # For any columns we want to "dollar‐weight," define a helper function:

    def dollar_weighted_avg(g, col):
        """
        Compute the dollar-weighted average of column `col`,
        weighting by (price * size) within the group g.
        """
        dollar_value = g["price"] * g["size"]
        numerator = (g[col] * dollar_value).sum()
        denominator = dollar_value.sum()
        return numerator / denominator if denominator != 0 else np.nan

    # Which columns from df_trade to compute a dollar‐weighted average for:
    dollar_weighted_cols = [
        "log_return",
        "trade_volume",
        "trade_impact_score",
        "realized_volatility",
    ]
    # (Note: Weighting "trade_intensity" by dollar value is unusual because
    #  "trade_intensity" is basically a count of trades per second.
    #  If you do want it, you can add it here.)

    # --------------------- #
    # 3. Timeframe Windows  #
    # --------------------- #

    # Original might be [150, 300, 450]. Just using [300] here:
    timeframes = [300]
    df_combined = []

    for timeframe in timeframes:
        order_filtered = df_order[df_order["seconds_in_bucket"] >= timeframe]
        trade_filtered = df_trade[df_trade["seconds_in_bucket"] >= timeframe]

        # 3A. Aggregate Order Features
        order_agg = order_filtered.groupby("time_id").agg(order_features).reset_index()

        # 3B. Aggregate Trade Features
        trade_agg = trade_filtered.groupby("time_id").agg(trade_features).reset_index()

        # 3C. Dollar Weighted WAP (as before)
        def dollar_wap(g):
            """Dollar-weighted price measure: sum(price^2*size)/sum(price*size)."""
            num = (g["price"] ** 2 * g["size"]).sum()
            den = (g["price"] * g["size"]).sum()
            return num / den if den != 0 else np.nan

        dollar_wap_df = (
            trade_filtered.groupby("time_id")
            .apply(dollar_wap)
            .reset_index(name="dollar_wap")
        )
        # Rename to e.g. 'dollar_wap_300'
        dollar_wap_df.rename(
            columns={"dollar_wap": f"dollar_wap_{timeframe}"}, inplace=True
        )

        # 3D. Dollar‐Weighted Averages for other columns
        # We'll compute them all in one pass, then pivot to a wide form:
        dw_avgs = (
            trade_filtered.groupby("time_id").apply(
                lambda g: {
                    f"{col}_dwavg_{timeframe}": dollar_weighted_avg(g, col)
                    for col in dollar_weighted_cols
                }
            )
        ).apply(
            pd.Series
        )  # Expand the dictionary into columns

        dw_avgs.reset_index(inplace=True)

        # flatten the columns
        trade_agg.columns = [
            "_".join(col) if isinstance(col, tuple) else col
            for col in trade_agg.columns
        ]
        trade_agg = trade_agg.rename(columns={"time_id_": "time_id"})

        # flatten the columns
        order_agg.columns = [
            "_".join(col) if isinstance(col, tuple) else col
            for col in order_agg.columns
        ]
        order_agg = order_agg.rename(columns={"time_id_": "time_id"})

        # 3E. Merge all trade data
        trade_agg = pd.merge(trade_agg, dollar_wap_df, on="time_id", how="left")
        trade_agg = pd.merge(trade_agg, dw_avgs, on="time_id", how="left")

        # 3F. Merge Order + Trade
        df_agg = pd.merge(
            order_agg,
            trade_agg,
            on="time_id",
            how="left",
            suffixes=(f"_order_{timeframe}", f"_trade_{timeframe}"),
        )
        df_combined.append(df_agg)

    # 4. Final Concatenation
    df_final = pd.concat(df_combined, axis=1)

    return df_final


if __name__ == "__main__":
    import pandas as pd
    from features import comprehensive_feature_engineering
    import warnings

    warnings.simplefilter(action="ignore")

    df_order = pd.read_pickle("output/order_book_sample.pkl")
    df_trade = pd.read_pickle("output/trade_sample.pkl")

    stock_id_lst = df_order.stock_id.unique()

    df_lst = []

    for stock_id in stock_id_lst:
        print(stock_id)
        df_order_stock = df_order[df_order.stock_id == stock_id]
        df_trade_stock = df_trade[df_trade.stock_id == stock_id]
        df = comprehensive_feature_engineering(df_order_stock, df_trade_stock)
        df["stock_id"] = stock_id
        df_lst.append(df)

    df = pd.concat(df_lst)
    df.to_pickle("output/computed_features_agg_150s.pkl")
