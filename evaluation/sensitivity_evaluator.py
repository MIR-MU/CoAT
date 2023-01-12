import abc
import random
from typing import Optional, Tuple, List, Union

from adaptor.evaluators.evaluator_base import EvaluatorBase
from adaptor.evaluators.generative import ROUGE
from transformers import PreTrainedTokenizer, PreTrainedModel

from evaluation.evaluator import Evaluator
from evaluation.tasks.task import Task


class InformativeEvaluatorBase(abc.ABC):

    def __init__(self,
                 task: Task,
                 num_demonstrations: int = 3,
                 firstn: Optional[int] = None,
                 bootstrap: bool = False,
                 max_input_length: Optional[int] = None,
                 reuse_last_run: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.num_demonstrations = num_demonstrations
        self.firstn = firstn
        self.bootstrap = bootstrap
        self.max_input_length = max_input_length
        self.reuse_last_run = reuse_last_run

    @abc.abstractmethod
    def _compute(self, expected: List[str], actual: List[str]) -> float:
        pass

    def _compute_bootstrapped(self,
                              expected_all: List[str],
                              actual_all: List[str],
                              per_round_samples: int = 100,
                              repeats: int = 200) -> List[float]:
        assert len(expected_all) == len(actual_all), "Prediction lists' length do not match"

        evals = []
        while len(evals) < repeats:
            subset_idx = [random.randrange(len(expected_all)) for _ in range(per_round_samples)]
            expected_subset = [expected_all[idx] for idx in subset_idx]
            actual_subset = [actual_all[idx] for idx in subset_idx]

            evals.append(self._compute(expected_subset, actual_subset))

        return evals

    def get_per_sampling_performance(self,
                                     model: PreTrainedModel,
                                     tokenizer: PreTrainedTokenizer,
                                     use_cache: bool = True) -> Tuple[Union[List[float], float],
                                                                      Union[List[float], float]]:
        # print("Model's performance in random selection: %s" % random_performance)
        # there's always less samples in 'informative' group
        expected, actual_informative, eval_set = Evaluator.collect_predictions(model, tokenizer, self.task,
                                                                               self.num_demonstrations, self.firstn,
                                                                               demo_selection_strategy="cluster-random",
                                                                               max_input_length=self.max_input_length,
                                                                               use_cache=use_cache)
        expected, actual_random, _ = Evaluator.collect_predictions(model, tokenizer, self.task,
                                                                   self.num_demonstrations, self.firstn,
                                                                   demo_selection_strategy="random",
                                                                   eval_set=eval_set,
                                                                   use_cache=use_cache)
        if self.bootstrap:
            informative_performance = self._compute_bootstrapped(expected, actual_informative)
            random_performance = self._compute_bootstrapped(expected, actual_informative)
        else:
            informative_performance = self._compute(expected, actual_informative)
            random_performance = self._compute(expected, actual_random)

        # print("Model's performance in informative selection: %s" % informative_performance)

        return random_performance, informative_performance

    def __str__(self):
        return "%s_%s" % (self.task, super().__str__())


class RougeRandom(InformativeEvaluatorBase, ROUGE):

    def _compute(self, expected: List[str], actual: List[str]) -> Union[float, List[float]]:
        return self.evaluate_str(expected, actual)

    def __call__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, _) -> float:
        random_perf_all, informative_perf_all = self.get_per_sampling_performance(model, tokenizer,
                                                                                  use_cache=self.reuse_last_run)

        if self.bootstrap:
            random_performance = sum(random_perf_all) / len(random_perf_all)  # average
        else:
            random_performance = random_perf_all

        return random_performance


class RougeInformative(RougeRandom, ROUGE):

    def __call__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, _) -> float:
        random_perf_all, informative_perf_all = self.get_per_sampling_performance(model, tokenizer,
                                                                                  use_cache=self.reuse_last_run)

        if self.bootstrap:
            informative_performance = sum(informative_perf_all) / len(informative_perf_all)  # average
        else:
            informative_performance = informative_perf_all

        return informative_performance


class AccuracyRandom(RougeRandom, EvaluatorBase):

    def _compute(self, expected: List[str], actual: List[str]) -> Union[float, List[float]]:
        num_correct = sum([exp == act for exp, act in zip(expected, actual)])
        return num_correct / len(expected)


class AccuracyInformative(AccuracyRandom, RougeInformative, EvaluatorBase):
    pass
