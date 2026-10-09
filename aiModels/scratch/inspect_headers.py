import rarfile
import zipfile
import pandas as pd

def inspect_sdalle():
    print("--- SDALLE Headers ---")
    try:
        with rarfile.RarFile("datasets/raw/SDALLE/Dataset_SDALLE.rar") as rf:
            csv_files = [f for f in rf.namelist() if f.endswith('.csv')]
            if csv_files:
                target = csv_files[0]
                print(f"Reading: {target}")
                with rf.open(target) as f:
                    df = pd.read_csv(f, nrows=2)
                    print(df.columns.tolist())
    except Exception as e:
        print("SDALLE Error:", e)

def inspect_mm_motion():
    print("\n--- MM-Motion Headers ---")
    try:
        with zipfile.ZipFile("datasets/raw/MM-Motion/subject01.zip") as zf:
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if csv_files:
                target = csv_files[0]
                print(f"Reading: {target}")
                with zf.open(target) as f:
                    df = pd.read_csv(f, nrows=2)
                    print(df.columns.tolist())
            else:
                print("No CSV files found in subject01.zip. Listing some files:")
                print(zf.namelist()[:10])
    except Exception as e:
        print("MM-Motion Error:", e)

if __name__ == "__main__":
    inspect_sdalle()
    inspect_mm_motion()
