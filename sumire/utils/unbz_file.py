import bz2
from pathlib import Path
from typing import Union


def unbz_file(input_filepath:Union[str, Path], output_filepath: Union[str, Path]):
    """
    Decompresses a BZ2-compressed file and saves the decompressed data to a new file.

    Args:
        input_filepath (str or Path): The path to the BZ2-compressed input file.
        output_filepath (str or Path): The path where the decompressed data will be saved.
    """
    with open(input_filepath, "rb") as source_file, open(output_filepath, "wb") as dest_file:
        decompressor = bz2.BZ2Decompressor()
        for data in iter(lambda: source_file.read(100 * 1024), b""):
            dest_file.write(decompressor.decompress(data))
