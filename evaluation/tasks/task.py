import abc
import os
from enum import Enum
from typing import Tuple, List, Union
from urllib.request import urlopen


class Metric(Enum):
    ACCURACY: int = 1
    FSCORE: int = 2


class Task(abc.ABC):

    label: str
    metric_type: int  # is Metric enum, but fails type check
    url: str
    data_file: str
    data: List[Tuple[str, str, str]] = []  # input, label, category
    template: Union[str, None] = None

    def __init__(self, cache_dir: str = ".") -> None:
        self.cache_dir = cache_dir
        if hasattr(self, "url"):
            self.data_file = self._maybe_download()

    def _maybe_download(self) -> str:
        fname = self.url.split("/")[-1]
        target_fpath = os.path.join(self.cache_dir, fname)
        if os.path.exists(target_fpath):
            return target_fpath
        else:
            print("Downloading %s" % self.url)
            resp = urlopen(self.url)
            with open(target_fpath, "wb") as data_f:
                data_f.write(resp.read())

            assert os.path.exists(target_fpath)

            return target_fpath

    def __str__(self) -> str:
        return self.__class__.__name__ + ("-%s" % str(self.template))
