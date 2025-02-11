import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = "model.keras"
NUM_SAMPLES_TO_CONSIDER = 22050  # 1 sec

class _Keyword_Spotting_Service:
    # tespit sistemin de tek seferde 1 tane anahtar kelime yüklemek istiyoruz. Atıyorum "cat" diyen bir ses ekleyeceğiz. Buna bakacak.

    model = None  # Buraya daha sonra bir tensorflow modeli ekleyeceğiz

    _mappings = [
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "four",
        "go",
        "happy",
        "house",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "wow",
        "yes",
        "zero"
    ]

    _instance = None  # Singleton tasarım deseni oluşturmak için gerekli olduğunu söylüyor


    def predict(self, file_path):
        # extract MFCCs
        MFCCs = self.preprocess(file_path)  # (# segments, # coefficients)

        # convert 2d MFCCs array into 4d array -> (#samples, #segments, #coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        # 2 boyutlu bir array'i 4 boyutlu bir array'e çevirmemiz gerekiyor. Çünkü CNN 4 boyutlu bir girdi bekliyor
        # İlk boyut number of samples

        # make prediction
        predictions = self.model.predict(MFCCs)  # [ [0.1, 0.6, 0.1, ...] ]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512 ):

        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T

def Keyword_Spotting_Service():
    # ensure that we only have 1 instance of KSS
    # anahtar kelime tespit hizmetinin KSS'inin yalnızca bir örneğine sahip olduğumuzdan emin olmak istiyoruz. Başka örnekleri olmasın

    if _Keyword_Spotting_Service._instance is None:
        # Eğer yoksa bir örnek oluştur
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        # En son eğittiğimiz modeli yükleyeceğiz
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

if __name__ == "__main__":

    kss = Keyword_Spotting_Service()

    keyword1 = kss.predict("test/dog.wav")
    keyword2 = kss.predict("test/sheila.wav")

    print(f"Predicted keyword: {keyword1}, {keyword2}")