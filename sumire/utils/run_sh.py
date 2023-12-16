from subprocess import run
from typing import Tuple, Union


def run_sh(command: str, return_err: bool = False) -> Union[str, Tuple[str, str]]:
    """
    Runs a shell command and captures its output.

    Args:
        command (str): The shell command to run.
        return_err (bool, optional): If True, returns both the standard output and standard error as a tuple.
            If False (default), returns only the standard output.

    Returns:
        str or Tuple[str, str]: The standard output of the command, or a tuple containing both standard output
        and standard error if `return_err` is True.
    """
    ret = run(command, shell=True, capture_output=True, text=True)
    if return_err:
        return ret.stdout, ret.stderr
    return ret.stdout
