import itertools
import os
import pickle
from typing import Iterable, Dict, List, Union, Optional, Tuple

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizer, PreTrainedModel

from common.demos_construction import selection_criterion, construct_sample
from evaluation import config
from evaluation.tasks.task import Task, Metric
import logging

logger = logging.getLogger()


class Evaluator:

    @staticmethod
    def evaluate(model: AutoModelForSeq2SeqLM,
                 tokenizer: PreTrainedTokenizer,
                 tasks: Iterable[Task]) -> Dict[str, float]:
        evaluations = {}

        for task in tasks:
            task_eval = Evaluator.evaluate_task(model, tokenizer, task)
            logger.warning("Task %s eval: %s" % (task, task_eval))

            evaluations[str(task)] = task_eval

        return evaluations

    @staticmethod
    def collect_predictions(model: PreTrainedModel,
                            tokenizer: PreTrainedTokenizer,
                            task: Task,
                            num_demonstrations: int,
                            firstn: Optional[int] = None,
                            demo_selection_strategy: str = config.demo_selection_strategy,
                            eval_set: Optional[List[Tuple[str, str, str]]] = None
                            ) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
        identifier = (str(model.name_or_path).split("/")[-1], task, demo_selection_strategy)

        cache_inputs_fpath = os.path.join(config.prediction_cache_dir,
                                          str(task) + "%s-%s-%s-inputs.txt" % identifier)
        cache_expected_fpath = os.path.join(config.prediction_cache_dir,
                                            str(task) + "%s-%s-%s-expected.txt" % identifier)
        cache_predicted_fpath = os.path.join(config.prediction_cache_dir,
                                             str(task) + "%s-%s-%s-predicted.txt" % identifier)
        if os.path.exists(cache_expected_fpath) and os.path.exists(cache_predicted_fpath):
            logger.warning("Reloading predictions for %s from %s, %s", task, cache_expected_fpath, cache_predicted_fpath)

            expected_texts = [l.strip() for l in open(cache_expected_fpath).readlines()]
            predicted_texts = [l.strip() for l in open(cache_predicted_fpath).readlines()]
            with open(cache_inputs_fpath, "rb") as out_f:
                eval_set_out = pickle.loads(out_f.read())
        else:
            if not os.path.exists(config.prediction_cache_dir):
                os.makedirs(config.prediction_cache_dir)

            expected_texts = []
            predicted_texts = []

            eval_set_in = task.data if eval_set is None else eval_set
            eval_set_out = []
            num_samples = firstn if firstn is not None else config.firstn \
                if config.firstn is not None else len(eval_set_in)

            skipped = 0
            for batch_offset in tqdm(range(0, num_samples, config.batch_size), desc="Evaluating %s" % task):
                tuples_batch = eval_set_in[batch_offset: batch_offset + config.batch_size]
                input_texts = []
                targets = []

                for sample in tuples_batch:
                    demonstrations = []
                    while len(demonstrations) < num_demonstrations:
                        try:
                            demonstrations.append(next(demo for demo in reversed(task.data)
                                                       if demo[0] != sample[0] and demo not in demonstrations
                                                       and selection_criterion(sample, demo, demo_selection_strategy)))
                        except StopIteration:
                            break
                    if not len(demonstrations) == num_demonstrations:
                        skipped += 1
                        continue
                    eval_set_out.append(sample)
                    input_texts.append(construct_sample(demonstrations, sample))
                    targets.append(sample[1])
                try:
                    encodings = tokenizer(input_texts, return_tensors="pt", padding=True).to(model.device)
                except IndexError:
                    # logger.warning("Skipping sample %s" % input_texts)
                    continue

                predictions = model.generate(**encodings)
                pred_batch = tokenizer.batch_decode(predictions, skip_special_tokens=True)

                expected_texts.extend(targets)
                predicted_texts.extend(pred_batch)

            logger.warning("%s: Skipped samples: %s out of total: %s" % (task, skipped, num_samples))
            logger.warning("Saving predictions for %s into %s, %s", task, cache_expected_fpath, cache_predicted_fpath)
            with open(cache_expected_fpath, "w") as out_f:
                out_f.writelines([t+"\n" for t in expected_texts])
            with open(cache_predicted_fpath, "w") as out_f:
                out_f.writelines([t+"\n" for t in predicted_texts])
            with open(cache_inputs_fpath, "wb") as out_f:
                out_f.write(pickle.dumps(eval_set_out))

        return expected_texts, predicted_texts, eval_set_out

    @staticmethod
    def evaluate_task(model: AutoModelForSeq2SeqLM,
                      tokenizer: PreTrainedTokenizer,
                      task: Task,
                      num_demonstrations: int = 3) -> float:

        expected_texts, predicted_texts = Evaluator.collect_predictions(model, tokenizer, task, num_demonstrations)
        # BoolQ responds consistently in Czech
        # print(self._evaluate_results_for_metric(["ano" if "es" in e else "ne" for e in expected_texts], predicted_texts,
        #                                         task.metric_type, ignore_casing=True))

        return Evaluator._evaluate_results_for_metric(expected_texts,
                                                      predicted_texts,
                                                      task.metric_type,
                                                      ignore_casing=True)

    @staticmethod
    def _evaluate_results_for_metric(expected: List[str],
                                     actual: List[str],
                                     metric: Union[Metric, int],
                                     ignore_casing: bool) -> float:
        assert len(expected) == len(actual), "Different size of expected and actual predictions :("
        if ignore_casing:
            expected = [e.lower() for e in expected]
            actual = [a.lower() for a in actual]

        if metric.value == Metric.ACCURACY.value:
            return sum(e == a for e, a in zip(expected, actual)) / len(expected)
        elif metric.value == Metric.FSCORE.value:
            # token-level F1-score, averaged over all samples:
            fscores = []
            for expected_one, actual_one in zip(expected, actual):
                expected_answers_set = set(itertools.chain(*[a.split() for a in expected_one]))
                actual_answer_set = actual_one.split()

                true_positives = sum(a_word in expected_answers_set for a_word in actual_answer_set)
                false_positives = sum(a_word not in expected_answers_set for a_word in actual_answer_set)
                false_negatives = sum(e_word not in actual_answer_set for e_word in expected_answers_set)

                fscores.append(true_positives / (true_positives + 0.5 * (false_positives + false_negatives)))

            return sum(fscores) / len(fscores)
        else:
            raise ValueError("Not implemented metric: %s" % metric)
