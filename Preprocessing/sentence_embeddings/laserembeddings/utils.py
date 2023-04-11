from io import TextIOBase, StringIO
import re
import sys
import os
import urllib.request

def adapt_bpe_codes(bpe_codes_f: TextIOBase) -> TextIOBase:
    """
    Converts fastBPE codes to subword_nmt BPE codes.

    Args:
        bpe_codes_f (TextIOBase): the text-mode file-like object of fastBPE codes
    Returns:
        TextIOBase: subword_nmt-compatible BPE codes as a text-mode file-like object
    """
    return StringIO(
        re.sub(r'^([^ ]+) ([^ ]+) ([^ ]+)$',
               r'\1 \2',
               bpe_codes_f.read(),
               flags=re.MULTILINE))


class sre_performance_patch:
    """
    Patch fixing https://bugs.python.org/issue37723 for Python 3.7 (<= 3.7.4)
    and Python 3.8 (<= 3.8.0 beta 3)
    """

    def __init__(self):
        self.sre_parse = None
        self.original_sre_parse_uniq = None

    def __enter__(self):
        #pylint: disable=import-outside-toplevel
        import sys

        if self.original_sre_parse_uniq is None and (
                0x03070000 <= sys.hexversion <= 0x030704f0
                or 0x03080000 <= sys.hexversion <= 0x030800b3):
            try:
                import sre_parse
                self.sre_parse = sre_parse
                #pylint: disable=protected-access
                self.original_sre_parse_uniq = sre_parse._uniq
                sre_parse._uniq = lambda x: list(dict.fromkeys(x))
            except (ImportError, AttributeError):
                self.sre_parse = None
                self.original_sre_parse_uniq = None

    def __exit__(self, type_, value, traceback):
        if self.sre_parse and self.original_sre_parse_uniq:
            #pylint: disable=protected-access
            self.sre_parse._uniq = self.original_sre_parse_uniq
            self.original_sre_parse_uniq = None

# model downloads
IS_WIN = os.name == 'nt'

def non_win_string(s):
    return s if not IS_WIN else ''

CONSOLE_CLEAR = non_win_string('\033[0;0m')
CONSOLE_BOLD = non_win_string('\033[0;1m')
CONSOLE_WAIT = non_win_string('⏳')
CONSOLE_DONE = non_win_string('✅')
CONSOLE_STARS = non_win_string('✨')
CONSOLE_ERROR = non_win_string('❌')


def download_file(url, dest):
    print(f'{CONSOLE_WAIT}   Downloading {url}...', end='')
    sys.stdout.flush()
    urllib.request.urlretrieve(url, dest)
    print(f'\r{CONSOLE_DONE}   Downloaded {url}    ')


def download_models(output_dir, version=2):
    assert version in [1, 2]
    print(f'Downloading models into {output_dir}')
    print('')

    if version == 1:
        download_file('https://dl.fbaipublicfiles.com/laser/models/93langs.fcodes',
                    os.path.join(output_dir, '93langs.fcodes'))
        download_file('https://dl.fbaipublicfiles.com/laser/models/93langs.fvocab',
                    os.path.join(output_dir, '93langs.fvocab'))
        download_file(
            'https://dl.fbaipublicfiles.com/laser/models/bilstm.93langs.2018-12-26.pt',
            os.path.join(output_dir, 'bilstm.93langs.2018-12-26.pt'))
    if version == 2:
        download_file(
            'https://dl.fbaipublicfiles.com/nllb/laser/laser2.pt',
            os.path.join(output_dir, 'laser2.pt'))
        download_file(
            'https://dl.fbaipublicfiles.com/nllb/laser/laser2.spm',
            os.path.join(output_dir, 'laser2.spm'))
        download_file(
            'https://dl.fbaipublicfiles.com/nllb/laser/laser2.cvocab',
            os.path.join(output_dir, 'laser2.cvocab'))

    print('')
    print(f'{CONSOLE_STARS} You\'re all set!')