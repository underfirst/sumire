import gzip
import tarfile
from pathlib import Path
from typing import Union


def untar_file(input_filepath:Union[str, Path], output_path: Union[str, Path]):
    """
    Decompresses a tar.gz file and saves the decompressed data to a new file.

    Args:
        input_filepath (str or Path): The path to the tar.gz input file.
        output_path (str or Path): The path where the decompressed data will be saved.
    """
    with tarfile.open(input_filepath, "r:gz") as tar:
        tar.extractall(path=output_path)
