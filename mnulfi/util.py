'''

some utility functions 

'''
import os


def dat_dir(): 
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
    )
