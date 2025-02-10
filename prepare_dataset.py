from librosa import load, feature
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"

# 1 sec worth of sound
SAMPLES_TO_CONSIDER = 22050


# Librosada ses dosyası yüklemek için kullandığımız varsayılan ayar. Librosa ile bir ses dosyası yüklediğimizde
# Librosa bunu sample_rate oranını kullanarak ortaya çıkartıyor(oynatıyor). Ve örnek sayısı bir saniyedeki örnek
# sayısıdır. Bu yüzden 22050(Nyquist) Veri setindeki dosyaların süresi 1 saniyedir bu da 22050'ye eşittir

# Bu veri setini ön işlemeye tabii tutmak için kullanacağımız fonksiyon Temelde fonksiyonun yapacağı şey bütün ses
# dosyalarının MFCC'ni çıkartıp JSON dosyasında saklamak Bu veriler daha sonra eğitim amaçı derin öğrenme modelimiz
# tarafından kullanılacak Bu verilerin hepsini ses işlemeye sokmak çok fazla vakit alır dolayısıyla bunu çevrimdışı
# yapmak daha iyidir. Sonra bu bilgileri bir JSON dosyasına kaydedin ve sonra bir training(eğitim) zamanında geri alın.
def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    # data dictionary (burada ses dosyalarından çıkardığımız tüm verileri depolayacağız)
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }
    #   "mappings": ["on", "off"],
    #    "labels": [0, 0, 1, 1, ...],
    #    "MFCCs": [],
    #    "files": ["dataset/on/1.wav"]

    # Listenin 0. ve 1. indeksine eşitlenecek on - off / Sinir ağımıza bir kelime geçiremiyoruz. Bunun yerine
    # sayıları geçirmeliyiz. Bu anahtar kelimeleri sayılara eşitlemeliyiz Sonra etiketleri(labels) yaratmak
    # istiyoruz. Sonra etiketlerin kendisi bir liste olacak ve burada [0, 0, 1, 1] gibi değerlere sahip olacağız.
    # Veri seti içindeki her ses dosyası için bir değere sahip olacağız. Örneğin 0 = "on" Örneğin 1 = "off" MFCCs'
    # çıktı olacak

    # files da "dataset/on/1.wav" diyerek ilk ses dosyasını analiz etmek için yolunu gösteriyoruz.
    # Bunu neden yapıyoruz? Ses dosyasına geri dönmek isteyebiliriz.

    # loop through all the sub-dirs - dirpath ana dizin- dirnames alt dizin(left gibi) - filenames zaten filenames
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # just we need to ensure that we're not at root level
        if dirpath is not dataset_path:

            # update mappings
            # category = dirpath.split("/")  # dataset/down -> [dataset, down]
            category = os.path.basename(dirpath)  # Doğru yöntem

            data["mappings"].append(category)
            print(f"Processing {category}")

            # loop through all the filenames and extract MFCCs
            for f in filenames:

                # get file path
                file_path = str(os.path.join(dirpath, f))

                # load audio file
                # we take signal and sample rate on librosa libs
                signal, sr = load(file_path)

                # ensure the audio file is at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # enforce 1 sec long signal (sinyal 1 saniyeden uzun olabilir bu durumda saniyelik kısmını alacağız
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract the MFCCs
                    MFCCs = feature.mfcc(y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

                    # store data (i-1 yapıyoruz çünkü dataset'te dönerken dataset'i = 0, down'u 1 alarak devam
                    # ediyor. Ama dataset aslında bir üst klasör. Biz iç klasörler ile ilgileniyoruz)
                    data["labels"].append(i-1)

                    # feature.mfcc'den elde ettiğimiz dizi bir numpy dizisidir.
                    data["MFCCs"].append(MFCCs.T.tolist()) #JSON dosyasında depolayabilmek için listeye dönüştürüyoruz. 

                    # dosya isimlerini kaydediyoruz. Dosya adını değil dosya yolunu girdik
                    data["files"].append(file_path)
                    print(f"{file_path}: {i-1}")

    # store in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4) # indent'i 4'e eşitiliyoruz böylece JSON dosyamız için güzel bir görüntü ortaya
        # çıkıyor.

if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)