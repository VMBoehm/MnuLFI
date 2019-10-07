import os

# Path to data files required for cosmos
_MNULFI_DATA_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

def dat_dir():
    return _MNULFI_DATA_DIR
