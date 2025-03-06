import pandas as pd, os

data_folder = (
    "/Users/kelvinyeung/PycharmProjects/optiver_volatility_prediction_kaggle/data"
)

# Define file paths
train_file_path = os.path.join(data_folder, "trade_train.parquet")
book_file_path = os.path.join(data_folder, "book_train.parquet")

# Load the data
trade_train_data = pd.read_parquet(train_file_path)
book_train_data = pd.read_parquet(book_file_path)

res = trade_train_data.groupby("stock_id")["size"].sum()
top_stocks = list(res.sort_values(ascending=False).head(15).index)
top_stocks.append(4)

book_train_data_sampled = book_train_data[book_train_data.stock_id.isin(top_stocks)]
trade_train_data_sampled = trade_train_data[trade_train_data.stock_id.isin(top_stocks)]

trade_train_data_sampled.to_pickle("data/trade_train_data_sampled.pkl")
book_train_data_sampled.to_pickle("data/book_train_data_sampled.pkl")
