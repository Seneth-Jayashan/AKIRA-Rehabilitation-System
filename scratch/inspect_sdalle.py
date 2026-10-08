import rarfile
try:
    with rarfile.RarFile("datasets/raw/SDALLE/Dataset_SDALLE.rar") as rf:
        print(rf.namelist()[:20])
except Exception as e:
    print("Error:", e)
