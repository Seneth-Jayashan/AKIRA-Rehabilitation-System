import pandas as pd
def load_standard_data(filepath):
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)
