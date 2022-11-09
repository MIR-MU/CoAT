import torch
from promptsource.templates import DatasetTemplates
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

from evaluation.sensitivity_evaluator import RougeInfoDIff, AccuracyInfoDIff

# TODO: aren't SQuAD examples more informative?
# dataset = load_dataset("squad")
from evaluation.tasks.en.glue_diagnostics import GLUEDiagnostics
# from evaluation.tasks.en.qa import AdversarialQATask
import argparse

from evaluation.tasks.en.openbookqa import OpenBookQATask
from evaluation.tasks.en.r4c_hotpotqa import R4CHotpotQATask

parser = argparse.ArgumentParser()

parser.add_argument("--model_names_or_paths", default="gaussalgo/mt5-base-priming-QA_en-cs", type=str,
                    help="Coma-separated list of evaluated models' identifiers")
parser.add_argument("--dataset_ids", default="glue/mnli", type=str,
                    help="Coma-separated list of evaluation datasets. Must be one of the implemented datasets: "
                         "'glue/mnli', 'openbookqa/additional', 'hotpot_qa/fullwiki'")
parser.add_argument("--template_names", default=None, type=str,
                    help="Names of the templates to evaluate with")
parser.add_argument("--metric", default="ROUGE", type=str,
                    help="A metric to compute informative difference with. Must be one of the implemented metrics:"
                         "'ROUGE', 'Accuracy'.")
parser.add_argument("--bootstrap", default=True, type=bool,
                    help="Whether to collect a set of results over random subsets of predictions. Defaults to True.")

args = parser.parse_args()
results = {}

max_memory_mapping = {0: "65GB", 1: "65GB"}

for model_name_or_path in args.model_names_or_paths.split(","):
    results[model_name_or_path] = {}
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path,
                                                      device_map="auto",
                                                      # max_memory=max_memory_mapping
                                                      )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                     device_map="auto",
                                                     max_memory=max_memory_mapping
                                                     )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    for dataset_id in args.dataset_ids.split(","):
        # eval templates resolution
        if args.template_names is not None:
            eval_templates = args.template_names
        else:
            if dataset_id == 'hotpot_qa/fullwiki':
                # only two templates for hotpot_qa require answering questions, others are for different tasks
                eval_templates = ['generate_answer_interrogative', 'generate_answer_affirmative']
            else:
                eval_templates = DatasetTemplates(dataset_id).all_template_names

        for template_id in eval_templates:
            template = DatasetTemplates(dataset_id)[template_id]
            # eval task resolution - done in the loop to reset its state (deduplication)
            if dataset_id == "glue/mnli":
                task = GLUEDiagnostics("en", template)
            elif dataset_id == "openbookqa/additional":
                task = OpenBookQATask("en", template)
            elif dataset_id == 'hotpot_qa/fullwiki':
                task = R4CHotpotQATask("en", template)
            else:
                raise ValueError("Non-implemented dataset: %s" % dataset_id)

            # evaluation metric resolution
            if args.metric == "ROUGE":
                evaluator = RougeInfoDIff(task, bootstrap=args.bootstrap)
            elif args.metric == "Accuracy":
                evaluator = AccuracyInfoDIff(task, bootstrap=args.bootstrap)
            else:
                raise ValueError("Unknown metric: %s" % args.metric)

            # a list of results if args.bootstrap, a single prediction otherwise
            random_selection_perf, info_selection_perf = evaluator(model, tokenizer, None)
            if not args.bootstrap:
                # unify the format, so we have a single result formatting
                random_selection_perf, info_selection_perf = [random_selection_perf], [info_selection_perf]

            for random_selection_perf_one, info_selection_perf_one in zip(random_selection_perf, info_selection_perf):
                print("{}\t{}\t{}\t{:.5f}\t{:.5f}\t{:.5f}".format(model_name_or_path,
                                                                  dataset_id,
                                                                  template_id,
                                                                  random_selection_perf_one,
                                                                  info_selection_perf_one,
                                                                  info_selection_perf_one - random_selection_perf_one))
                results[model_name_or_path][template_id] = {"random": random_selection_perf_one,
                                                            "info": info_selection_perf_one,
                                                            "diff": info_selection_perf_one - random_selection_perf_one}

# print(results)
