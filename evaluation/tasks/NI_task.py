from typing import Tuple, List

from evaluation.tasks.task import Task


class NITask(Task):

    def __init__(self, demonstrations: List[Tuple[str, str]], label: str):
        super().__init__()
        self.data = [(d[0], d[1], None) for d in demonstrations]  # type: ignore
        self.label = label
