import os
import zipfile
import subprocess
import glob


def inspect_zip(zip_path, max_files=10):
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            names = z.namelist()
            print(f"--- ZIP: {zip_path} ({len(names)} files) ---")
            print("Sample files:", names[:max_files])
            
            # find first csv or txt
            for name in names:
                if name.endswith('.csv') or name.endswith('.txt'):
                    print(f"Reading sample from {name}:")
                    with z.open(name) as f:
                        lines = [f.readline().decode('utf-8', errors='ignore').strip() for _ in range(5)]
                        for l in lines:
                            print(l)
                    break
    except Exception as e:
        print(f"Failed to inspect zip {zip_path}: {e}")

def inspect_dir(dir_path):
    print(f"--- DIR: {dir_path} ---")
    files = []
    for root, dirs, filenames in os.walk(dir_path):
        for name in filenames:
            files.append(os.path.join(root, name))
            if len(files) > 10:
                break
        if len(files) > 10:
            break
    print("Sample files:", [os.path.relpath(f, dir_path) for f in files[:10]])
    for f in files:
        if f.endswith('.csv') or f.endswith('.txt'):
            print(f"Reading sample from {f}:")
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                    for _ in range(5):
                        print(file.readline().strip())
            except Exception as e:
                print("Failed to read", e)
            break

def inspect_rar(rar_path):
    print(f"--- RAR: {rar_path} ---")
    try:
        # list files
        out = subprocess.check_output(['unrar', 'lb', rar_path], text=True)
        files = out.split('\n')
        print("Sample files:", files[:10])
        # read first csv/txt
        for f in files:
            if f.endswith('.csv') or f.endswith('.txt'):
                print(f"Reading sample from {f}:")
                content = subprocess.check_output(['unrar', 'p', '-inul', rar_path, f], text=True)
                lines = content.split('\n')[:5]
                for l in lines:
                    print(l)
                break
    except Exception as e:
        print(f"Failed to inspect rar {rar_path}: {e}")

print("INSPECTING DATASETS...\n")

print("\n--- GAITEX ---")
inspect_zip('datasets/raw/GAITEX/data.zip')

print("\n--- SDALLE ---")
inspect_rar('datasets/raw/SDALLE/Dataset_SDALLE.rar')

print("\n--- MM-Motion ---")
inspect_zip('datasets/raw/MM-Motion/subject01.zip')

print("\n--- PHYTMO ---")
inspect_dir('datasets/raw/PHYTMO')

print("\n--- AnkleImage ---")
inspect_dir('datasets/raw/AnkleImage/Task 1')

