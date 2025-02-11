"""
Flask sunucusunu çalıştırıcağız. Localhost'ta 5000 numaralı portu dinleyeceğiz.
Ardından yapmak istediğimiz şey, istemciden bir HTTP POST isteği göndermek. Bu gönderi isteği sunucuya ulaşacak.
Sunucu bu isteği okuyacak. ve gönderi isteğiyle birlikte paketlenecek. ses dosyasını çıkaracak ve ardından keyword_spotting_service
ses dosyasını tahmin edecek.

Bu yapı temelde inşa edeceğimiz her şey
client -> POST request -> server -> prediction back to client

"""
import random
import os
from flask import Flask, request, jsonify
from keyword_spotting_service import Keyword_Spotting_Service
from waitress import serve

app = Flask(__name__)

# Aslında yapmak istediğimiz şey gelen bir isteği bir API uç noktasına yönlendirmek.
# HTTP, POST ya da GET request'leri gelir. Gelen istedikleri farklı view'lere yönlendirir.

# Gelen bütün istekleri işleyecek bir fonksiyon yazıyoruz.
"""
Alan adı gibi istek gönderdiğimizde, bu uygulama için örneğin "ks.com/predict" gönderildiğinde Flask ne yapar?
 Bu request'i alır ve aşağıdaki görünme yönlendirir. Aşağıdaki "predict" metodunda post isteğiyle birlikte paketlenmiş 
 ses dosyasını almamız gerekiyor. Sonra speech recognition sistemimizi ya da Tensorflow modelini örneklendirmek gerekiyor
 Sonra tahmini yapmak ve tahmini istemciye geri göndermek istiyoruz.
"""


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Received request...")  # Debugging step

        # Check if 'audio_file' exists in the request
        if "audio_file" not in request.files:
            print("No file part in request")
            return jsonify({"error": "No file uploaded"}), 400

        # Save the audio file
        audio_file = request.files["audio_file"]
        file_name = f"temp_{random.randint(0, 100000)}.wav"
        audio_file.save(file_name)
        print(f"Saved file: {file_name}")

        # invoke(uyandırmak-çağırmak) keyword spotting service
        kss = Keyword_Spotting_Service()

        # make a prediction (ses dosyasında hangi anahtar kelimenin bulundupunu tespit edebilmek için tahmin yap)
        predicted_keyword = kss.predict(file_name)

        # remove the audio file (Geçici olarak depoladığımız ses dosyasını kaldıracağız)
        os.remove(file_name)
        print(f"Removed file: {file_name}")

        # Return JSON response
        response = jsonify({"keyword": predicted_keyword})
        print("Sending response:", response.get_json())
        return response

    except Exception as e:
        print(f"Error occurred: {e}")  # Print error message
        return jsonify({"error": str(e)}), 500  # Return error response

if __name__ == "__main__":
    #app.run(debug=False)
    print("Starting server on http://127.0.0.1:5050")
    serve(app, host="127.0.0.1", port=5050, threads=4)
