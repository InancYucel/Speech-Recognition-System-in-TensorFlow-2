import requests

URL = 'http://127.0.0.1:5050/predict'
TEST_AUDIO_FILE_PATH = "test/dog.wav"

if __name__ == "__main__":
    file = open(TEST_AUDIO_FILE_PATH, "rb")

    # package stuff to send and perform POST request
    values = {"audio_file": (TEST_AUDIO_FILE_PATH, file, "audio/wav")}
    # API uç noktasına bir POST isteği gönderip değerleri atar
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted keyword is: {data['keyword']}")
