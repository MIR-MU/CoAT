import torch
from promptsource.templates import DatasetTemplates
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

from evaluation.sensitivity_evaluator import RougeInformative, AccuracyInformative

# TODO: aren't SQuAD examples more informative?
# dataset = load_dataset("squad")
from evaluation.tasks.en.glue_diagnostics import GLUEDiagnostics
# from evaluation.tasks.en.qa import AdversarialQATask
import argparse

from evaluation.tasks.en.openbookqa import OpenBookQATask
from evaluation.tasks.en.r4c_hotpotqa import R4CHotpotQATask
from evaluation.tasks.en.worldtree_qa import WorldTreeQA

parser = argparse.ArgumentParser()

parser.add_argument("--model_names_or_paths", default="gaussalgo/mt5-base-priming-QA_en-cs", type=str,
                    help="Coma-separated list of evaluated models' identifiers")
parser.add_argument("--dataset_ids", default="glue/mnli", type=str,
                    help="Coma-separated list of evaluation datasets. Must be one of the implemented datasets: "
                         "'glue/mnli', 'openbookqa/additional', 'hotpot_qa/fullwiki', 'worldtree'")
parser.add_argument("--template_names", default=None, type=str,
                    help="Names of the templates to evaluate with")
parser.add_argument("--metric", default="ROUGE", type=str,
                    help="A metric to compute informative difference with. Must be one of the implemented metrics:"
                         "'ROUGE', 'Accuracy'.")
parser.add_argument("--bootstrap", default=True, type=bool,
                    help="Whether to collect a set of results over random subsets of predictions. Defaults to True.")
parser.add_argument("--max_input_length", default=None, type=int,
                    help="Maximum in-context input size, in a number of space-separated words.")
parser.add_argument("--num_demonstrations", default=3, type=int,
                    help="Number of in-context task demonstrations to prepend.")
parser.add_argument("--firstn", type=int, default=0,
                    help="If given, a number of samples from dataset to evaluate on.")

args = parser.parse_args()
args.use_cache = args.use_cache == "True"
args.bootstrap = args.bootstrap == "True"
results = {}

max_memory_mapping = {0: "45GB", 1: "65GB", 2: "65GB", 3: "55GB"}

for model_name_or_path in args.model_names_or_paths.split(","):
    results[model_name_or_path] = {}
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path,
                                                      # device_map=device_map,
                                                      device_map="auto",
                                                      # max_memory=max_memory_mapping
                                                      )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    for dataset_id in args.dataset_ids.split(","):
        # eval templates resolution
        if args.template_names is not None:
            eval_templates = args.template_names.split(",")
        else:
            if dataset_id == 'hotpot_qa/fullwiki':
                # only two templates for hotpot_qa require answering questions, others are for different tasks
                eval_templates = ['generate_answer_interrogative', 'generate_answer_affirmative']
            else:
                eval_templates = DatasetTemplates(dataset_id).all_template_names
                if not eval_templates:
                    eval_templates = ["no template"]

        for template_id in eval_templates:
            template = DatasetTemplates(dataset_id)[template_id] if template_id != "no template" else None
            # eval task resolution - done in the loop to reset its state (deduplication)
            if dataset_id == "glue/mnli":
                task = GLUEDiagnostics("en", template)
            elif dataset_id == "openbookqa/additional":
                task = OpenBookQATask("en", template)
            elif dataset_id == 'hotpot_qa/fullwiki':
                task = R4CHotpotQATask("en", template)
            elif dataset_id == 'worldtree':
                task = WorldTreeQA("en", template)
            else:
                raise ValueError("Non-implemented dataset: %s" % dataset_id)

            # evaluation metric resolution
            if args.metric == "ROUGE":
                evaluator = RougeInformative(task,
                                             bootstrap=args.bootstrap,
                                             max_input_length=args.max_input_length,
                                             firstn=args.firstn if args.firstn else None,
                                             num_demonstrations=args.num_demonstrations)
            elif args.metric == "Accuracy":
                evaluator = AccuracyInformative(task,
                                                bootstrap=args.bootstrap,
                                                max_input_length=args.max_input_length,
                                                num_demonstrations=args.num_demonstrations)
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
                                                                  dataset_id,
                                                                  template_id,
                                                                  random_selection_perf_one,
                                                                  info_selection_perf_one,
                                                                  info_selection_perf_one - random_selection_perf_one))
                results[model_name_or_path][template_id] = {"random": random_selection_perf_one,
                                                            "info": info_selection_perf_one,
                                                            "diff": info_selection_perf_one - random_selection_perf_one}

# print(results)
