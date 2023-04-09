import os
import urllib.request

from Utility.storage_config import MODELS_DIR


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
    os.makedirs(os.path.join(MODELS_DIR, "Aligner"), exist_ok=True)
    filename, headers = urllib.request.urlretrieve(
        url="https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/aligner.pt",
        filename=os.path.abspath(os.path.join(MODELS_DIR, "Aligner", "aligner.pt")),
        reporthook=report)

    #############
    print("Downloading Multilingual ToucanTTS Model")
    os.makedirs(os.path.join(MODELS_DIR, "ToucanTTS_Meta"), exist_ok=True)
    filename, headers = urllib.request.urlretrieve(
        url="https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/ToucanTTS_Meta.pt",
        filename=os.path.abspath(os.path.join(MODELS_DIR, "ToucanTTS_Meta", "best.pt")),
        reporthook=report)

    #############
    print("Downloading Fast Vocoder Model")
    os.makedirs(os.path.join(MODELS_DIR, "Avocodo"), exist_ok=True)
    filename, headers = urllib.request.urlretrieve(
        url="https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/Avocodo.pt",
        filename=os.path.abspath(os.path.join(MODELS_DIR, "Avocodo", "best.pt")),
        reporthook=report)

    #############
    print("Downloading Highest Quality Vocoder Model")
    os.makedirs(os.path.join(MODELS_DIR, "BigVGAN"), exist_ok=True)
    filename, headers = urllib.request.urlretrieve(
        url="https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/BigVGAN.pt",
        filename=os.path.abspath(os.path.join(MODELS_DIR, "BigVGAN", "best.pt")),
        reporthook=report)

    #############
    print("Downloading Embedding Model")
    os.makedirs(os.path.join(MODELS_DIR, "Embedding"), exist_ok=True)
    filename, headers = urllib.request.urlretrieve(
        url="https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/embedding_function.pt",
        filename=os.path.abspath(os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt")),
        reporthook=report)

    #############
    print("Downloading Embedding GAN")
    os.makedirs(os.path.join(MODELS_DIR, "Embedding"), exist_ok=True)
    filename, headers = urllib.request.urlretrieve(
        url="https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/embedding_gan.pt",
        filename=os.path.abspath(os.path.join(MODELS_DIR, "Embedding", "embedding_gan.pt")),
        reporthook=report)


if __name__ == '__main__':
    download_models()
