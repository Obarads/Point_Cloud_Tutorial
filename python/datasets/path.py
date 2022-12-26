import os, sys

def add_dir_path():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../"))) # for package path
