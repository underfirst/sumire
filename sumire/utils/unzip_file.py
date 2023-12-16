import os
import zipfile
from pathlib import Path
from typing import Union


def unzip_file(file_path: Union[str, Path], dir_path: Union[str, Path]):
    """
    Unzips a file to a specified directory.

    Args:
        file_path (str or Path): The path to the ZIP file to be extracted.
        dir_path (str or Path): The directory where the contents of the ZIP file will be extracted.
    """
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        zip_ref.extractall(dir_path)
