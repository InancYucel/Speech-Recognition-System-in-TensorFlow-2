import json
import numpy as np
from sklearn.model_selection import train_test_split
# sklearn python'daki klasik makine öğrenmesi kütüphanesidir.

import tensorflow as tf
from tensorflow import keras

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.keras"
LEARNING_RATE = 0.0001

NUM_KEYWORDS = 30  # datasetimizde 30 farklı kelime bulunuyor bu yüzden 30

# EPOCHS sayısı bize eğitim amaçlı tüm veri setine kaç kere gittiğini söyler
EPOCHS = 40

# İş boyutu. Ağın bir update'den önce göreceği örnek sayısı belirtir ve geri yayımlım algoritması gibi çalışır.
# Learning SAP gibi çalışır
BATCH_SIZE = 32

def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

        # extract inputs and targets
        X = np.array(data["MFCCs"])
        y = np.array(data["labels"])

        # Girişleri ve hedefleri doğru bir şekilde çıkarmamız gerekiyor.

        return X, y


def get_data_splits(data_path, test_size=0.1, test_validation=0.1):
    # load the datasets
    # Burada X tüm girdiler - y ise tüm etiketler veya beklediğimiz çıktı hedefleri
    X, y = load_dataset(data_path)

    # create train/validation/test splits

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # test_size şu demek veri kümesinden test kümesine katılacak olanların oranı train_test_split fonksiyonunu input
    # değelerini bu şekil "X_train, X_test, y_train, y_test" output değerlerine ayırmak için kullanıyoruz. Temelde
    # veri setinin %90'ı eğitim ve doğrulamaya gidiyor

    # train validation ve test splits girdilerini görmemiz gerekiyor. İki boyutlu diziler olacak çünkü bunlar JSON
    # dosyasından geliyor. Temelde bunlar MFCC değerleri bu yüzden burada iki boyutlu dizileri, serbest boyutlu
    # dizilere dönüştürmemiz gerekiyor.

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_validation)

    # convert inputs from 2d to 3d

    # (# segments, 13, 1) 13=MFCC katsayısı? İlk boyut segments - İkinci boyut 13 - Üçüncü boyut 1
    # (...) "bana dizideki şu ana kadar olan tüm şeyleri ver anlamına geliyor.
    # np.newaxis kullanarak yeni bir boyut ekleyin
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # sonra da tüm veri bölümlerini döndürebiliriz.
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):
    # build network sequential - ardışık bir model oluşturmak istiyoruz. Bir sürü ardışık katmana sahip bir ayna
    # ağıdır. Biri diğerine, yenisi üçüncüsüne etc
    model = keras.Sequential()

    # conv layer 1
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    # eklediğimiz layer bir conv layerdır. Parametre 1 = Filtre / Kernel_Size / activation fonksiyonu
    # relu meaning = düzeltilmiş doğrusal birimler
    # overfitting için kernel_regularizer kullanıyor. Bunun için 14 numaralı bir videosu var.

    # bu conv layerdan sonra batch_normalization yapan başka bir katman ekleyeceğiz.
    # batch_normalization eğitimi hızlandırmak ve daha iyi sonuçlar elde etmek için kullandığımız başka bir tekniktir.
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

    # conv layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

    # conv layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu", input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # serbest boyutlu olan yukarıdaki çıktıları alıp buraya vereceğiz. Neden? Tek boyutlu bir diziye çevirmek istiyoruz.
    # Neden? Çünkü dense layers(yoğun katmanların) girdi olarak 1 boyutlu bir diziye ihtiyacı var.
    # flatten the output feed it into a dense layer
    model.add(keras.layers.Flatten())

    # 1. parametre number of units ya da neuron
    model.add(keras.layers.Dense(64, activation="relu"))

    # overfitting'i ele almak için başka bir teknik.Dropout rate - Bırakma oranı. Eğitim sırasında dense layerdaki
    # neuronların %30'unu kapatmak. Bu fonksiyonu AI'a sor. Bunu yaparsak anlyıoruz ki ağın sınıflandırma yapmak için
    # mutlaka bir veya birkaç nörona güvenmemesi, bunun yerine ağın uyum sağlaması ve tüm farklı neuronların
    # sınıflandırmanın kendisi için eşit sorumluluk alması gerekliliğidir.
    model.add(keras.layers.Dropout(0.3))

    # softmax classifier
    # Çıktı katmanı
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation="softmax"))
    # 10 farklı keyword'ümüz olduğunu biliyoruz fakat basitlik için 3 tane var olarak düşünelim
    # [0.1, 0.7, 0.2]

    # compile the model
    # sonraki adım modeli derlemek

    optimiser = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=error, metrics=["accuracy"])
    # metrics accuracy izlemek istediğimiz ölçümleri belirtiyoruz. Eğitim sırasında doğruluğunu kontrol edeceğiz.

    # print model overview
    model.summary()

    return model

def main():

    print("TensorFlow Version:", tf.__version__)
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))
    print("Is built with CUDA:", tf.test.is_built_with_cuda())

    # TensorFlow'un hangi cihazları kullandığını görmek için
    print("Logical Devices:", tf.config.list_logical_devices())

    with open("data.json", "r") as f:
        try:
            data = json.load(f)
            print("JSON düzgün formatlanmış.")
        except json.JSONDecodeError as e:
            print(f"JSON hatalı: {e}")

    # Load train/validation/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    # Build the CNN models CNN (Evrimsel Sinir Ağı), özellikle görüntü işleme ve bilgisayarla görme (computer vision)
    # alanlarında kullanılan bir derin öğrenme modelidir. Geleneksel yapay sinir ağlarından farklı olarak,
    # CNN'ler uzaysal verileri (örneğin, görüntüleri) doğrudan işleyebilmek için özel katmanlar kullanır

    # Burada ne yaptığımızı anlayalım. 3 Boyutlu bir diziye ihtiyacımız var nedeni üç boyutlu girdiler alan bir
    # evrişimli sinir ağını kullanmamız." Bahsedilen şey şu ""segment sayısına göre verilir" prepare_dataset.py
    # kısmında eşit aralıklı segmentlerde MFCC'leri çıkardığımızı hatırlarsınız. Bir segmentin uzunluğu ses
    # dosyalarımızda bulunan sample_rate veya benzer örneklerin toplam miktarının hop_length'e bölünmesiyle elde edildi.
    # bu da bize sahip olduğumuz segment sayısını verecek
    # Bizim case'imizde coefficients(katsayılar matrisi) 13' eşitti. Bu yüzden 13 MFCC çıkarttık
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # (# segments, # coefficients 13, 1)
    # 3 boyut bir CNN temelidir. 3 boyutlu olmasının sebebi bir görüntünün derinliği ve sözde kanalı hakkında bilgi
    # taşıyan boyut olmasıdır. Örneğin gri scale(tonlamalı) bir görüntüde derinlik 1'e eşittir.Eğer RGB görüntümüz
    # olsaydı bu kanal 3'e eşit olurdu. Audio data'larıyla uğraştığımızı ve MFCC'nin özelliklerini göz önünde
    # bulundurarak sanki gri tonlamalı bir görüntü gibi bazı verilerle uğraşıyormuşuz gibi. Bu yüzden derinlik 1'e
    # eşit. Yani bir kanal bizim için iyi.

    model = build_model(input_shape, LEARNING_RATE)

    # build network

    # compile the model

    # train the model

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_validation, y_validation))
    # Modelimizi eğitmek için direkt Keras API'ından gelen ve Fit olarak adlandırılan bir yöntemi
    # kullanabileceğimizi böylece bir model nokta uyumu yaptığımızı ve burada ihtiyacımız olan bir sürü farklı
    # argümanı geçirin, bu yüzden her şeyden önce eğitim verilerini gerçimemiz gerekecek.(x_train, y_train)
    # Hem girdileri hem de çıktıları geçireceğiz. EPOCHS kaç dönem geçmesini istediğimizi belirtmemiz gerekiyor.
    # epochs = EPOCHS
    # Temel olarak model.fit'i bu şekilde kullanarak konuşma tanıma sistemimizi eğiteceğiz.

    # evaluate the model

    # Değerlendirme sürecinden bir test_error alacağız, ayrıca test_accuracy de.

    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}, Test accuracy: {test_accuracy}")
    # Bu şekilde modeli değerlendirebiliriz

    # save the model
    model.save(SAVED_MODEL_PATH)

if __name__ == "__main__":
    main()