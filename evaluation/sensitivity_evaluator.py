from typing import Optional, Tuple

from adaptor.evaluators.generative import ROUGE
from transformers import PreTrainedTokenizer, PreTrainedModel

from evaluation.evaluator import Evaluator
from evaluation.tasks.task import Task


class RougeInfoDIff(ROUGE):

    def __init__(self, task: Task,
                 num_demonstrations: int = 3, firstn: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.num_demonstrations = num_demonstrations
        self.firstn = firstn

    def __call__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, _) -> Tuple[float, float]:
        expected, actual_random = Evaluator.collect_predictions(model, tokenizer,
                                                                self.task, self.num_demonstrations, self.firstn,
                                                                demo_selection_strategy="random")
        random_performance = self.evaluate_str(expected, actual_random)
        # print("Model's performance in random selection: %s" % random_performance)

        expected, actual_informative = Evaluator.collect_predictions(model, tokenizer,
                                                                     self.task, self.num_demonstrations, self.firstn,
                                                                     demo_selection_strategy="cluster-random")
        informative_performance = self.evaluate_str(expected, actual_informative)

        # print("Model's performance in informative selection: %s" % informative_performance)

        return random_performance, informative_performance

    def __str__(self):
        return "%s_%s" % (self.task, super().__str__())
