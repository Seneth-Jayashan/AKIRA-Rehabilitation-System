import pandas as pd
import numpy as np

def run_data_quality_check(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    missing = df.isnull().sum().sum()
    dups = df.duplicated().sum()
    
    report = f"# Model Input Quality Report\n\n"
    report += f"- Numeric columns: {len(numeric_cols)}\n"
    report += f"- Categorical columns: {len(categorical_cols)}\n"
    report += f"- Missing values: {missing}\n"
    report += f"- Duplicate rows: {dups}\n"
    report += f"- Participants: {df.get('participant_id', pd.Series()).nunique()}\n"
    report += f"- Trials: {df.get('trial_id', pd.Series()).nunique()}\n\n"
    
    report += "## Ranges and Outliers\n"
    for col in numeric_cols:
        if col not in ['trial_id', 'segment_id'] and col in df.columns:
            report += f"- **{col}**: Min = {df[col].min():.4f}, Max = {df[col].max():.4f}, Mean = {df[col].mean():.4f}\n"
            
    with open('08_RESULTS/model_input_quality_report.md', 'w') as out:
        out.write(report)
    return df
