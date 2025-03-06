import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class NearestNeighborFeatureGeneratorNoPivot:
    """
    Demonstration of generating stock-id and time-id NN features
    without using pivot tables.

    For each (stock_id, time_id), we compute aggregated metrics from
    df_order and df_trade. Then we do a KNN among those rows to find
    neighbors in feature space. The same logic is repeated for time-id
    features. In a real pipeline, you'd enforce the rule that for time_id = N,
    you only use data up to time_id < N for your aggregations.
    """

    def __init__(
        self, df_order: pd.DataFrame, df_trade: pd.DataFrame, n_neighbors: int = 5
    ):
        self.df_order = df_order.copy()
        self.df_trade = df_trade.copy()
        self.n_neighbors = n_neighbors

    def _compute_order_features(self):
        """
        Example aggregator for order-book data, at (stock_id, time_id) level.
        In a real approach, you'd only use time_id < current row's time_id
        for a cumulative or rolling aggregator. Here we do a simple direct groupby.
        """
        df = self.df_order.copy()

        # Basic calculations (same as original code snippet)
        df["bid_ask_spread"] = df["ask_price1"] - df["bid_price1"]
        df["wap1"] = (
            df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]
        ) / (df["bid_size1"] + df["ask_size1"] + 1e-6)

        df["order_imbalance"] = (df["bid_size1"] - df["ask_size1"]) / (
            df["bid_size1"] + df["ask_size1"] + 1e-6
        )

        # Example aggregator: average across all seconds_in_bucket for each (stock_id, time_id)
        agg_order = (
            df.groupby(["stock_id", "time_id"])
            .agg(
                {
                    "bid_ask_spread": "mean",
                    "wap1": "mean",
                    "order_imbalance": "mean",
                }
            )
            .reset_index()
        )
        agg_order = agg_order[agg_order.stock_id.isin(df["stock_id"].unique())]

        return agg_order

    def _compute_trade_features(self):
        """
        Example aggregator for trade data, at (stock_id, time_id) level.
        """
        df = self.df_trade.copy()

        # Suppose we define "trade_volume" = sum of sizes, "trade_impact" = sum of size*price
        # for each (stock_id, time_id).
        df["trade_impact"] = df["size"] * df["price"]

        agg_trade = (
            df.groupby(["stock_id", "time_id"])
            .agg(
                {
                    "size": "sum",  # total volume
                    "trade_impact": "sum",  # sum of (size*price)
                }
            )
            .reset_index()
        )
        agg_trade = agg_trade[agg_trade.stock_id.isin(df["stock_id"].unique())]
        agg_trade = agg_trade.rename(columns={"size": "trade_volume"})
        return agg_trade

    def _merge_order_trade(self):
        """
        Merge order-features and trade-features into a single
        DataFrame with columns: [stock_id, time_id, bid_ask_spread,
        wap1, order_imbalance, trade_volume, trade_impact].
        """
        order_agg = self._compute_order_features()
        trade_agg = self._compute_trade_features()

        # TODO change this
        df_merged = pd.merge(
            order_agg, trade_agg, on=["stock_id", "time_id"], how="left"
        ).fillna(0)

        return df_merged

    def generate_stock_id_nn_features(self):
        """
        Demonstrate 'stock_id nearest neighbors' by looking at
        each time_id in turn, computing KNN across the (stock_id) dimension
        for that same time_id. Then we create aggregated neighbor features.
        """
        df_agg = self._merge_order_trade()

        # We will store the final neighbor-based features for each row in a list
        all_results = []

        # Feature columns to use in the KNN distance
        feature_cols = [
            "bid_ask_spread",
            "wap1",
            "order_imbalance",
            "trade_volume",
            "trade_impact",
        ]

        # Sort by time_id so we can simulate "use only up to time_id < current" if needed
        unique_time_ids = sorted(df_agg["time_id"].unique())

        for t in unique_time_ids:
            # TODO time_id filter. sort list of time ids first
            # Here, we have a single "snapshot" of how each stock looks at time_id = t
            # If you want to exclude t itself and only use t-1, you'd do something like:
            # historical_data = df_agg[df_agg["time_id"] < t]
            # But typically, we want to do KNN among stocks *at the same time t*:
            current_data = df_agg[df_agg["time_id"] == t].copy()

            # If there are fewer stocks than n_neighbors, skip
            if len(current_data) < 2:
                # no neighbors possible, or trivial
                current_data = current_data.assign(
                    **{f"nn_stock_ids": [[]] * len(current_data)}
                )
                all_results.append(current_data)
                continue

            # Fit a NearestNeighbors model on the feature vectors
            X = current_data[feature_cols].values
            nn_model = NearestNeighbors(
                n_neighbors=min(self.n_neighbors + 1, len(X)), metric="euclidean"
            )
            nn_model.fit(X)
            distances, indices = nn_model.kneighbors(X)

            # For each row, store the stock_ids of its neighbors, excluding itself
            current_data["nn_stock_ids"] = [
                current_data.iloc[indices[i][1:]]["stock_id"].tolist()
                for i in range(len(indices))
            ]

            all_results.append(current_data)

        df_stock_nn = pd.concat(all_results, ignore_index=True).sort_values(
            ["time_id", "stock_id"]
        )
        return df_stock_nn

    def generate_time_id_nn_features(self):
        """
        Demonstrate 'time_id nearest neighbors' by computing metrics aggregated
        across all stocks for each time_id, then using a KNN among time_ids in that
        aggregated space.
        """

        # Example: compute some time-level features from the trade data
        df = self.df_trade.copy()

        # Suppose each row has: (stock_id, time_id, price, size, order_count, ...)
        # We'll define a few "time-level" aggregates across all stocks for each time_id:
        df["value_traded"] = df["price"] * df["size"]

        # We'll group by time_id alone (aggregating across all stocks)
        time_agg = df.groupby("time_id", as_index=False).agg(
            {
                "size": "sum",  # total size across all stocks
                "value_traded": "sum",  # total value traded
                "order_count": "sum",  # total number of trades
            }
        )

        # Example: define "rolling_volatility" across all stocks as well
        # We'll do a quick example: volatility of 'price' across all rows in that time_id
        # In reality, you'd do something more sophisticated or handle each time separately.
        def volatility_of_prices(subdf):
            # if there's only one row, volatility is 0
            if len(subdf) < 2:
                return 0.0
            p = subdf["price"].values
            return np.std(np.diff(np.log(p + 1e-9)))

        vol_df = (
            self.df_trade.groupby("time_id").apply(volatility_of_prices).reset_index()
        )
        vol_df.columns = ["time_id", "rolling_volatility"]

        # TODO change this fillna
        # Merge the volatility into time_agg
        time_agg = pd.merge(time_agg, vol_df, on="time_id", how="left").fillna(0)

        # Now we do a KNN among these time_id rows
        feature_cols = ["size", "value_traded", "order_count", "rolling_volatility"]
        X = time_agg[feature_cols].values

        if len(time_agg) < 2:
            # No neighbors to compute, just attach empty columns
            time_agg["nn_time_ids"] = [[]] * len(time_agg)
            return time_agg

        nn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, len(X)), metric="euclidean"
        )
        nn_model.fit(X)
        distances, indices = nn_model.kneighbors(X)

        # Store the time_ids of neighbors, excluding itself
        time_agg["nn_time_ids"] = [
            time_agg.iloc[indices[i][1:]]["time_id"].tolist()
            for i in range(len(indices))
        ]

        return time_agg


def main():
    # # Example DataFrame
    # # Suppose each row in df_order is (time_id, stock_id, various order-book columns)
    # df_order = pd.DataFrame(
    #     {
    #         "time_id": np.arange(10),
    #         "seconds_in_bucket": np.random.randint(1, 100, size=10),
    #         "bid_price1": np.random.rand(10),
    #         "ask_price1": np.random.rand(10),
    #         "bid_price2": np.random.rand(10),
    #         "ask_price2": np.random.rand(10),
    #         "bid_size1": np.random.rand(10),
    #         "ask_size1": np.random.rand(10),
    #         "bid_size2": np.random.rand(10),
    #         "ask_size2": np.random.rand(10),
    #         "stock_id": np.random.randint(1, 5, size=10),
    #     }
    # )
    #
    # # Suppose each row in df_trade is (time_id, stock_id, price, size, order_count, ...)
    # df_trade = pd.DataFrame(
    #     {
    #         "time_id": np.arange(10),
    #         "seconds_in_bucket": np.random.randint(1, 100, size=10),
    #         "price": np.random.rand(10),
    #         "size": np.random.rand(10),
    #         "order_count": np.random.randint(1, 10, size=10),
    #         "stock_id": np.random.randint(1, 5, size=10),
    #     }
    # )

    df_order = pd.read_pickle("output/order_book_sample.pkl")
    df_order = df_order[df_order.time_id.isin(df_order.time_id.unique()[:3])]

    df_trade = pd.read_pickle("output/trade_sample.pkl")
    df_trade = df_trade[df_trade.time_id.isin(df_trade.time_id.unique()[:3])]

    nn_generator = NearestNeighborFeatureGeneratorNoPivot(
        df_order, df_trade, n_neighbors=3
    )

    # --- STOCK-ID NEIGHBORS ---
    df_stock_nn = nn_generator.generate_stock_id_nn_features()
    print("Stock-ID Nearest Neighbor Features:")
    print(df_stock_nn)

    # --- TIME-ID NEIGHBORS ---
    df_time_nn = nn_generator.generate_time_id_nn_features()
    print("\nTime-ID Nearest Neighbor Features:")
    print(df_time_nn)


if __name__ == "__main__":
    main()
