import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class NearestNeighborFeatureGenerator:
    def __init__(
        self, df_order: pd.DataFrame, df_trade: pd.DataFrame, n_neighbors: int = 5
    ):
        self.df_order = df_order
        self.df_trade = df_trade
        self.n_neighbors = n_neighbors

    def calculate_similarity(
        self, df: pd.DataFrame, feature_cols: list, pivot_col: str
    ):
        # Ensure feature_cols is a list of strings
        if isinstance(feature_cols, str):
            feature_cols = [feature_cols]

        # Group by 'stock_id' and 'time_id' and calculate mean for each feature
        grouped_df = (
            df.groupby(["stock_id", "time_id"])[feature_cols].mean().reset_index()
        )

        # Pivot the data to create a matrix for nearest neighbor calculation
        pivot_table = grouped_df.pivot(
            index="time_id", columns="stock_id", values=feature_cols
        )

        # Flatten the column MultiIndex to ensure unique column names
        pivot_table.columns = [
            f"{feature}_{stock_id}" for feature, stock_id in pivot_table.columns
        ]

        # Fill missing values with the mean of each column
        pivot_table = pivot_table.fillna(pivot_table.mean())

        # Fit the NearestNeighbors model
        nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine")
        nn_model.fit(pivot_table)

        # Find the nearest neighbors
        distances, indices = nn_model.kneighbors(pivot_table)

        return pivot_table, indices

    def aggregate_features(self, pivot_table, indices, feature_cols: list):
        # Aggregate features from nearest neighbors
        nn_features = {}
        for feature_col in feature_cols:
            nn_features.update(
                {
                    f"{feature_col}_nn_mean": np.mean(
                        pivot_table[feature_col].values[indices], axis=1
                    ),
                    f"{feature_col}_nn_min": np.min(
                        pivot_table[feature_col].values[indices], axis=1
                    ),
                    f"{feature_col}_nn_max": np.max(
                        pivot_table[feature_col].values[indices], axis=1
                    ),
                    f"{feature_col}_nn_std": np.std(
                        pivot_table[feature_col].values[indices], axis=1
                    ),
                }
            )
        return pd.DataFrame(nn_features, index=pivot_table.index)

    def create_stock_id_features(self):
        # Calculate necessary features from df_order
        self.df_order["bid_ask_spread"] = (
            self.df_order["ask_price1"] - self.df_order["bid_price1"]
        )
        self.df_order["wap1"] = (
            self.df_order["bid_price1"] * self.df_order["ask_size1"]
            + self.df_order["ask_price1"] * self.df_order["bid_size1"]
        ) / (self.df_order["bid_size1"] + self.df_order["ask_size1"])
        self.df_order["order_imbalance"] = (
            self.df_order["bid_size1"] - self.df_order["ask_size1"]
        ) / (self.df_order["bid_size1"] + self.df_order["ask_size1"])

        # Calculate trade features
        self.df_trade["trade_volume"] = self.df_trade.groupby("stock_id")[
            "size"
        ].transform("sum")
        self.df_trade["trade_impact"] = self.df_trade["size"] * self.df_trade["price"]

        # Merge trade features into order DataFrame
        df_combined = pd.merge(
            self.df_order,
            self.df_trade[["stock_id", "trade_volume", "trade_impact"]],
            on="stock_id",
            how="left",
        )

        # Features for stock_id nearest neighbors
        stock_features = [
            "bid_ask_spread",
            "wap1",
            "order_imbalance",
            "trade_volume",
            "trade_impact",
        ]
        pivot_table, indices = self.calculate_similarity(
            df_combined, stock_features, "stock_id"
        )
        return self.aggregate_features(pivot_table, indices, stock_features)

    def create_time_id_features(self):
        # Calculate necessary features from df_trade
        self.df_trade["aggregate_liquidity"] = self.df_trade["size"] / (
            self.df_trade["order_count"] + 1
        )
        self.df_trade["market_pressure"] = (
            self.df_trade["size"] * self.df_trade["price"]
        )
        self.df_trade["market_activity"] = self.df_trade["size"]
        self.df_trade["rolling_volatility"] = self.df_trade.groupby("time_id")[
            "price"
        ].transform(lambda x: np.std(np.log(x / x.shift(1))))

        # Features for time_id nearest neighbors
        time_features = [
            "aggregate_liquidity",
            "market_pressure",
            "market_activity",
            "rolling_volatility",
        ]
        pivot_table, indices = self.calculate_similarity(
            self.df_trade, time_features, "time_id"
        )
        return self.aggregate_features(pivot_table, indices, time_features)

    def generate_features(self):
        stock_id_features = self.create_stock_id_features()
        time_id_features = self.create_time_id_features()
        return stock_id_features, time_id_features


def main():
    # Example DataFrame
    df_order = pd.DataFrame(
        {
            "time_id": np.arange(10),
            "seconds_in_bucket": np.random.randint(1, 100, size=10),
            "bid_price1": np.random.rand(10),
            "ask_price1": np.random.rand(10),
            "bid_price2": np.random.rand(10),
            "ask_price2": np.random.rand(10),
            "bid_size1": np.random.rand(10),
            "ask_size1": np.random.rand(10),
            "bid_size2": np.random.rand(10),
            "ask_size2": np.random.rand(10),
            "stock_id": np.random.randint(1, 5, size=10),
        }
    )

    df_trade = pd.DataFrame(
        {
            "time_id": np.arange(10),
            "seconds_in_bucket": np.random.randint(1, 100, size=10),
            "price": np.random.rand(10),
            "size": np.random.rand(10),
            "order_count": np.random.randint(1, 10, size=10),
            "stock_id": np.random.randint(1, 5, size=10),
        }
    )

    nn_generator = NearestNeighborFeatureGenerator(df_order, df_trade, n_neighbors=3)
    stock_id_features, time_id_features = nn_generator.generate_features()

    print("Stock ID Nearest Neighbor Features:")
    print(stock_id_features)
    print("\nTime ID Nearest Neighbor Features:")
    print(time_id_features)


if __name__ == "__main__":
    main()
