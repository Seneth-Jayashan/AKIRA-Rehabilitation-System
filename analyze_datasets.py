import os
import csv
import json

datasets = {
    "3DCDataset": "3DCDataset",
    "HandArmRehab": "HandArmRehab",
    "OpenMuscle": "OpenMuscle",
    "29666735": "29666735_Dataset",
    "WatchHand": "WatchHand"
}

def analyze_dataset(name, path):
    print(f"\n=======================================================")
    print(f"DATASET: {name}")
    print(f"=======================================================")
    
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return
        
    file_types = {}
    csv_headers = {}
    
    total_files = 0
    total_size = 0
    
    for root, _, files in os.walk(path):
        for file in files:
            total_files += 1
            ext = os.path.splitext(file)[1].lower()
            file_types[ext] = file_types.get(ext, 0) + 1
            
            filepath = os.path.join(root, file)
            try:
                total_size += os.path.getsize(filepath)
            except Exception:
                pass
                
            if ext in ['.csv', '.txt', '.tsv'] and len(csv_headers) < 10:
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        if ext == '.csv':
                            reader = csv.reader(f)
                            header = next(reader)
                        elif ext == '.tsv':
                            reader = csv.reader(f, delimiter='\t')
                            header = next(reader)
                        else:
                            header = f.readline().strip().split()
                        
                        # Just store unique headers up to 10 distinct ones to sample
                        header_tuple = tuple(header[:20]) # Only first 20 columns
                        if header_tuple not in csv_headers.values():
                            csv_headers[file] = header_tuple
                except Exception as e:
                    pass
                    
    print(f"Total files: {total_files}")
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    print("File types:", file_types)
    print("\nSample Headers:")
    for f, h in csv_headers.items():
        print(f"  {f}: {h}")

for name, path in datasets.items():
    analyze_dataset(name, path)
