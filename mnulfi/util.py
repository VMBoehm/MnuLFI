'''

some utility functions 

'''
import os


def check_env(): 
    if os.environ.get('MNULFI_DIR') is None: 
        raise ValueError("set $MNULFI_DIR in bashrc file!") 
    return None


def dat_dir(): 
    return os.environ.get('MNULFI_DIR') 

