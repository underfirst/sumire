from os import makedirs, remove
from pathlib import Path
from typing import Union

import requests
from loguru import logger
from tqdm import tqdm


def download_file(url: str, filepath: Union[str, Path], timeout: int = 600):
    """
    Downloads a file from a URL and saves it to a specified filepath.
    If download time out or the status code is not 200,
    just put error log message, but it never raises Exception to continue program.

    Args:
        url (str): The URL to download the file from.
        filepath (str or Path): The path where the downloaded file should be saved.
        timeout (int, optional): The maximum time (in seconds) to wait for the download. Default is 600 seconds.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    ret = requests.get(url, stream=True, timeout=timeout)

    if ret.status_code == 200:
        total_size = int(ret.headers.get("content-length", 0))
        block_size = 1024  # 1 KB


        makedirs(filepath.parent, exist_ok=True)
        with open(filepath, "wb") as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
        ) as pbar:
            for data in ret.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)
    else:
        logger.error(f"Failed to download from {url}. Status code: {ret.status_code}")
        remove(filepath)
