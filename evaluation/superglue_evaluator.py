import argparse

import pandas as pd
from promptsource.templates import DatasetTemplates
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from evaluation.sensitivity_evaluator import RougeInformative, \
    AccuracyInformative
from evaluation.tasks.en.superglue import all_task_classes

parser = argparse.ArgumentParser()

parser.add_argument("--model_names_or_paths", default="gaussalgo/mt5-base-priming-QA_en-cs", type=str,
                    help="Coma-separated list of evaluated models' identifiers")
parser.add_argument("--use_cache", type=str, default="True", choices=('True', 'False'),
                    help="Whether to use cached predictions, if available.")
parser.add_argument("--firstn", type=int, default=500,
                    help="If given, a number of samples from dataset to evaluate on.")
parser.add_argument("--metric", default="ROUGE", type=str,
                    help="A metric to compute informative difference with. Must be one of the implemented metrics:"
                         "'ROUGE', 'Accuracy'.")
parser.add_argument("--max_input_length", default=None, type=int,
                    help="Whether to collect a set of results over random subsets of predictions. Defaults to True.")
parser.add_argument("--bootstrap", default=False, type=bool,
                    help="Whether to collect a set of results over random subsets of predictions. Defaults to True.")

args = parser.parse_args()

results = {}

# eval iteration
for model_name_or_path in args.model_names_or_paths.split(","):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path,
                                                  # device_map=device_map,
                                                  # device_map="auto",  # TODO
                                                  # max_memory=max_memory_mapping
                                                  )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    results[model_name_or_path] = {}
    for SGLUETaskClass in all_task_classes:
        for template_name in DatasetTemplates(SGLUETaskClass.promptsource_id).all_template_names:
            task = SGLUETaskClass(prompts_template=template_name)

            if args.metric == "ROUGE":
                evaluator = RougeInformative(task,
                                             bootstrap=args.bootstrap,
                                             max_input_length=args.max_input_length,
                                             firstn=args.firstn if args.firstn else None)
            elif args.metric == "Accuracy":
                evaluator = AccuracyInformative(task, bootstrap=args.bootstrap, max_input_length=args.max_input_length)
            else:
                raise ValueError("Unknown metric: %s" % args.metric)

            # a list of results if args.bootstrap, a single prediction otherwise
            random_selection_perf, info_selection_perf = evaluator.get_per_sampling_performance(model, tokenizer,
                                                                                                args.use_cache)
            if not args.bootstrap:
                # unify the format, so we have a single result formatting
                random_selection_perf, info_selection_perf = [random_selection_perf], [info_selection_perf]

            for random_selection_perf_one, info_selection_perf_one in zip(random_selection_perf, info_selection_perf):
                print("{}\t{}\t{}\t{:.5f}\t{:.5f}\t{:.5f}".format(model_name_or_path,
                                                                  task.promptsource_id,
                                                                  template_name,
                                                                  random_selection_perf_one,
                                                                  info_selection_perf_one,
                                                                  info_selection_perf_one - random_selection_perf_one))
                result_key = "%s-%s" % (task.promptsource_id, template_name)
                results[model_name_or_path][result_key] = {"random": random_selection_perf_one,
                                                           "info": info_selection_perf_one,
                                                           "diff": info_selection_perf_one - random_selection_perf_one}

    pd.DataFrame(results).to_csv("%s_superglue_evaluation.tsv" % model_name_or_path, sep="\t")
