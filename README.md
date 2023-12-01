![image](Utility/toucan.png)

IMS Toucan is a toolkit for teaching, training and using state-of-the-art Speech Synthesis models, developed at the
**Institute for Natural Language Processing (IMS), University of Stuttgart, Germany**. Everything is pure Python and
PyTorch based to keep it as simple and beginner-friendly, yet powerful as possible.

This branch is specifically made with minimal overhead to generate audios for the [ASVSpoof Challenge](https://www.asvspoof.org/)

---

## Installation 🦉

These instructions should work for most cases, but I heard of some instances where espeak behaves weird, which are
sometimes resolved after a re-install and sometimes not. Also, M1 and M2 MacBooks require a very different installation
process, with which I am unfortunately not familiar.

#### Basic Requirements

To install this toolkit, clone it onto the machine you want to use it on
(should have at least one GPU if you intend to train models on that machine. For
inference, you don't need a GPU).
Navigate to the directory you have cloned. We recommend creating and activating a
[virtual environment](https://docs.python.org/3/library/venv.html)
to install the basic requirements into. The commands below summarize everything
you need to do under Linux. If you are running Windows, the second line needs
to be changed, please have a look at the
[venv documentation](https://docs.python.org/3/library/venv.html).

```
python -m venv <path_to_where_you_want_your_env_to_be>

source <path_to_where_you_want_your_env_to_be>/bin/activate

pip install --no-cache-dir -r requirements.txt
```

Run the second line everytime you start using the tool again to activate the virtual environment again, if you e.g.
logged out in the meantime. To make use of a GPU, you don't need to do anything else on a Linux machine. On a Windows
machine, have a look at [the official PyTorch website](https://pytorch.org/) for the install-command that enables GPU
support.

#### espeak-ng

And finally you need to have espeak-ng installed on your system, because it is used as backend for the phonemizer. If
you replace the phonemizer, you don't need it. On most Linux environments it will be installed already, and if it is
not, and you have the sufficient rights, you can install it by simply running

```
apt-get install espeak-ng
```

For other systems, e.g. Windows, they provide a convenient .msi installer file
[on their GitHub release page](https://github.com/espeak-ng/espeak-ng/releases). After installation on non-linux
systems, you'll also need to tell the phonemizer library where to find your espeak installation, which is discussed in
[this issue](https://github.com/bootphon/phonemizer/issues/44#issuecomment-1008449718). Since the project is still in
active development, there are frequent updates, which can actually benefit your use significantly.

#### Storage configuration

If you don't want the pretrained and trained models as well as the cache files resulting from preprocessing your
datasets to be stored in the default subfolders, you can set corresponding directories globally by
editing `Utility/storage_config.py` to suit your needs (the path can be relative to the repository root directory or
absolute).

#### Pretrained Models

Run the `run_model_downloader.py` script to automatically download them from the release page and put
them into their appropriate locations with appropriate names.