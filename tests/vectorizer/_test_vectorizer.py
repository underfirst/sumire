from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from sumire.vectorizer.base.common import BaseVectorizer

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data/test/test.txt"

test_lines = list(open(TEST_DATA_DIR).readlines())


def _test_vectorizer(vectorizer: BaseVectorizer, head: int = -1):
    with TemporaryDirectory() as dir_path:
        # dir_path = TEST_DATA_DIR.parent
        dir_path = Path(dir_path)
        vectorizer.fit(test_lines[:head])
        ret = vectorizer.fit_transform(test_lines[:head])

        vectorizer.save_pretrained(dir_path)
        loaded = vectorizer.from_pretrained(dir_path)
        loaded_ret = loaded.transform(test_lines[:head])
        assert np.array_equal(ret, loaded_ret)
