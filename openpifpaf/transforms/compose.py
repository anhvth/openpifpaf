from typing import List

from .preprocess import Preprocess


class Compose(Preprocess):
    """Execute given transforms in sequential order."""
    def __init__(self, preprocess_list: List[Preprocess]):
        self.preprocess_list = preprocess_list

    def __call__(self, *args):
        for trans in self.preprocess_list:
            if trans is None:
                continue
            args = trans(*args)

        return args
