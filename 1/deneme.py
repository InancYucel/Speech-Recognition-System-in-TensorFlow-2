from librosa import load, feature
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"

# 1 sec worth of sound
SAMPLES_TO_CONSIDER = 22050

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    # data dictionary (burada ses dosyalarından çıkardığımız tüm verileri depolayacağız)
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:

            # Hatalı yöntem (Linux için çalışabilir ama Windows için sıkıntılı olabilir)
            category1 = dirpath.split("/")[-1]

            # Doğru yöntem (Platformdan bağımsızdır)
            category2 = os.path.basename(dirpath)

            print(f"split yöntemi: {category1}")
            print(f"os.path.basename yöntemi: {category2}")

if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)