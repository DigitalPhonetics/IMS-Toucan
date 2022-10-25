import os
import urllib.request


def report(block_number, read_size, total_size):
    if block_number % 1000 == 0:
        return_to_front = '\b' * 52
        percent = round(((block_number * read_size) / total_size) * 100)
        print(f"{return_to_front}[{'█' * (percent // 2)}{'.' * (50 - (percent // 2))}]", end='')
    if block_number * read_size >= total_size:
        return_to_front = '\b' * 52
        print(f"{return_to_front}Download complete!\n")


def download_models():
    #############
    print("Downloading Aligner Model")
    os.makedirs("Models/Aligner", exist_ok=True)
    filename, headers = urllib.request.urlretrieve(
        url="https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.3/aligner.pt",
        filename=os.path.abspath("./Models/Aligner/aligner.pt"),
        reporthook=report)

    #############
    print("Downloading Multilingual FastSpeech 2 Model")
    os.makedirs("Models/FastSpeech2_Meta", exist_ok=True)
    filename, headers = urllib.request.urlretrieve(
        url="https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.3/FastSpeech2_Meta.pt",
        filename=os.path.abspath("./Models/FastSpeech2_Meta/best.pt"),
        reporthook=report)

    #############
    print("Downloading Vocoder Model")
    os.makedirs("Models/Avocodo", exist_ok=True)
    filename, headers = urllib.request.urlretrieve(
        url="https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.3/Avocodo.pt",
        filename=os.path.abspath("./Models/Avocodo/best.pt"),
        reporthook=report)

    #############
    print("Downloading Embedding Model")
    os.makedirs("Models/Embedding", exist_ok=True)
    filename, headers = urllib.request.urlretrieve(
        url="https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.3/embedding_function.pt",
        filename=os.path.abspath("./Models/Embedding/embedding_function.pt"),
        reporthook=report)

    #############
    print("Downloading Embedding GAN")
    os.makedirs("Models/Embedding", exist_ok=True)
    filename, headers = urllib.request.urlretrieve(
        url="https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.3/embedding_gan.pt",
        filename=os.path.abspath("./Models/Embedding/embedding_gan.pt"),
        reporthook=report)


if __name__ == '__main__':
    download_models()
