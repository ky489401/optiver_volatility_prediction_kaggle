# This script processes financial data to reconstruct the order of time IDs using t-SNE for dimensionality reduction.
# The goal is to treat each time_id as a point in a high-dimensional space, where temporally close time_ids should be close in that space.
# By applying t-SNE, we aim to find a continuous 1D manifold that represents the temporal order of time_ids.

# Import necessary libraries
import glob
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib.parallel import delayed
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale


# Define a context manager for timing code execution
@contextmanager
def timer(name: str):
    """Context manager to measure the execution time of a code block."""
    start_time = time.time()  # Record the start time
    yield  # Execute the block of code within the 'with' statement
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(
        f"[{name}] {elapsed_time: .3f}sec"
    )  # Print the elapsed time with the given name


# Function to calculate prices from parquet files
def calc_prices(r):
    """Calculate prices from a parquet file for a given stock."""
    # Read specific columns from the parquet file
    df = pd.read_parquet(
        r.book_path,
        columns=["time_id", "ask_price1", "ask_price2", "bid_price1", "bid_price2"],
    )
    df = df.set_index("time_id")  # Set 'time_id' as the index
    # Apply 'calc_price2' function to each group of 'time_id' and convert to DataFrame
    df = df.groupby(level="time_id").apply(calc_price2).to_frame("price").reset_index()
    df["stock_id"] = r.stock_id  # Add 'stock_id' to the DataFrame
    return df  # Return the DataFrame with calculated prices


# Function to calculate price tick
def calc_price2(df):
    """Calculate the price tick from the DataFrame."""
    # Calculate the smallest difference between unique sorted prices
    tick = sorted(np.diff(sorted(np.unique(df.values.flatten()))))[0]
    return 0.01 / tick  # Return the inverse of the tick size


# Function to sort data using t-SNE
def sort_manifold(df, clf):
    """Sort the DataFrame using t-SNE for dimensionality reduction."""
    df_ = df.set_index("time_id")  # Set 'time_id' as the index
    # Scale the data to a range between 0 and 1, filling NaNs with column means
    df_ = pd.DataFrame(minmax_scale(df_.fillna(df_.mean())))

    # Fit the t-SNE model and transform the data
    X_compoents = clf.fit_transform(df_)

    # Reorder the DataFrame based on the first component of t-SNE
    dft = df.reindex(np.argsort(X_compoents[:, 0])).reset_index(drop=True)
    return np.argsort(X_compoents[:, 0]), X_compoents  # Return the order and components


# Main function to reconstruct the order of time IDs
def reconstruct_time_id_order():
    """Reconstruct the order of time IDs using t-SNE."""
    with timer("load files"):
        # Load parquet file paths and extract 'stock_id' from the file path
        df_files = pd.DataFrame(
            {"book_path": glob.glob("./data/book_train.parquet/**/*.parquet")}
        ).eval(
            'stock_id = book_path.str.extract("stock_id=(\d+)").astype("int")',
            engine="python",
        )

    with timer("calc prices"):
        # Calculate prices in parallel for each file and concatenate the results
        df_prices = pd.concat(
            Parallel(n_jobs=4, verbose=51)(
                delayed(calc_prices)(r) for _, r in df_files.iterrows()
            )
        )
        # Pivot the DataFrame to have 'time_id' as index and 'stock_id' as columns
        df_prices = df_prices.pivot_table(
            index="time_id", columns="stock_id", values="price"
        )
        # Rename columns to include 'stock_id=' prefix
        df_prices.columns = [f"stock_id={i}" for i in df_prices.columns]
        df_prices = df_prices.reset_index(
            drop=False
        )  # Reset index to include 'time_id'
        df_prices.to_pickle("df_prices.pkl")  # Save the DataFrame to a pickle file

    with timer("t-SNE(400) -> 50"):
        # Perform t-SNE with a high perplexity to get initial components
        clf = TSNE(n_components=1, perplexity=400, random_state=0, n_iter=2000)
        order, X_compoents = sort_manifold(df_prices, clf)

        # Perform t-SNE again with a lower perplexity using the initial components
        clf = TSNE(
            n_components=1,
            perplexity=50,
            random_state=0,
            init=X_compoents,
            n_iter=2000,
            method="exact",
        )
        order, X_compoents = sort_manifold(df_prices, clf)

        # Reorder the DataFrame based on the t-SNE order
        df_ordered = df_prices.reindex(order).reset_index(drop=True)
        # Reverse the order if the first value is greater than the last for 'stock_id=61'
        if df_ordered["stock_id=61"].iloc[0] > df_ordered["stock_id=61"].iloc[-1]:
            df_ordered = df_ordered.reindex(df_ordered.index[::-1]).reset_index(
                drop=True
            )

    # Plot the reordered prices for 'stock_id=61'
    plt.plot(df_ordered["stock_id=61"])
    df_ordered.to_pickle("output/df_ordered.pkl")  # Save the ordered DataFrame

    return df_ordered[["time_id"]]  # Return the DataFrame with reordered 'time_id'


if __name__ == "__main__":
    # Execute the main function and print the first few rows of the result
    time_id_order = reconstruct_time_id_order()
    print(time_id_order.head())
