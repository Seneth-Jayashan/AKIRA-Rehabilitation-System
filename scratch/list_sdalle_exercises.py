import rarfile
import os

try:
    exercises = set()
    with rarfile.RarFile("datasets/raw/SDALLE/Dataset_SDALLE.rar") as rf:
        for name in rf.namelist():
            parts = name.split('/')
            if len(parts) > 1:
                exercises.add(parts[1])
    print("SDALLE Exercises found:")
    print(list(exercises))
except Exception as e:
    print("Error:", e)
