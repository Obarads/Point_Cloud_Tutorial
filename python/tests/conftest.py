import os, sys
import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../")))  # for package path

DATA_DIR_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../data"))


@pytest.fixture
def data_dir_path() -> str:
    return DATA_DIR_PATH
