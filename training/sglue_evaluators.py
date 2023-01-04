from typing import Optional

import torch
from adaptor.evaluators.generative import ROUGE
from transformers import PreTrainedTokenizer

from evaluation.evaluator import Evaluator
from evaluation.tasks.task import Task


class TaskROUGE(ROUGE):

    def __init__(self, task: Task, num_demonstrations: int = 3, firstn: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.num_demonstrations = num_demonstrations
        self.firstn = firstn

    def __call__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, _) -> float:
        expected, actual, demonstrations = Evaluator.collect_predictions(model,
                                                                         tokenizer,
                                                                         self.task, self.num_demonstrations,
                                                                         self.firstn,
                                                                         use_cache=False)
        return self.evaluate_str(expected, actual)

    def __str__(self):
        return "%s_%s" % (self.task, super().__str__())
