from typing import List, Dict

import pandas as pd
from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset

from evaluation.sensitivity_evaluator import RougeInformative, RougeRandom
from evaluation.tasks.en.adversarialqa import AdversarialQATask
from evaluation.tasks.en.glue_diagnostics import GLUEDiagnostics
from evaluation.tasks.en.r4c_hotpotqa import R4CHotpotQATask
from evaluation.tasks.en.superglue import all_task_classes
from evaluation.tasks.en.worldtree_qa import WorldTreeQA
from priming_objective import Priming
from training.sglue_evaluators import TaskROUGE

training_arguments = AdaptationArguments(output_dir="train_dir_teabreac+qa_hard_large",
                                         learning_rate=2e-5,  # we set LR=2e-4 for pre-training experiments
                                         # stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=100000,
                                         gradient_accumulation_steps=30,  # TODO: set
                                         eval_steps=100,  # TODO: set
                                         logging_steps=10,
                                         save_steps=1000,
                                         num_train_epochs=2,
                                         evaluation_strategy="steps",
                                         save_total_limit=10,
                                         stopping_patience=30)
eval_examples = 200  # TODO set

# priming
num_demonstrations = 3


def _construct_priming_prompt(previous_examples: List[str], current_example: str) -> str:
    return " ".join(previous_examples + [current_example])


# lang_module = LangModule("google/mt5-small")  # TODO set
# lang_module = LangModule("gaussalgo/mt5-base-priming-QA_en-cs")
# lang_module = LangModule("google/mt5-base")
lang_module = LangModule("google/mt5-large")

# priming
per_type_examples = {}

qa_en = load_dataset("adversarial_qa", "adversarialQA")
qa_train = qa_en["train"].filter(lambda entry: len(entry["context"]) < 2000)

qa_task = AdversarialQATask("en")
qa_diff_evaluator = RougeInformative(qa_task)

glue_task = GLUEDiagnostics("en")
hotpotqa = R4CHotpotQATask("en")
worldtree = WorldTreeQA("en")

concept_aware_metrics = [RougeInformative(qa_task),
                         RougeInformative(glue_task), RougeRandom(glue_task),
                         RougeInformative(hotpotqa), RougeRandom(hotpotqa),
                         RougeInformative(worldtree), RougeRandom(worldtree)]

val_metrics = [BLEU(**{"additional_sep_char": "▁"})]

superglue_metrics = [TaskROUGE(TaskCls(), num_demonstrations, firstn=eval_examples // 3) for TaskCls in
                     all_task_classes]


# Adversarial QA dataset & objective:

def _get_en_squad_categories(data) -> List[str]:
    return [question.split()[0] if not question.startswith("To")
            else " ".join(question.split()[:2])
            for question in data["question"]]


q_answering_en = Priming(lang_module,
                         difficulty_sample=5,  # TODO set
                         demos_selection_strategy="hard",  # TODO set
                         texts_or_path=qa_train["question"],
                         text_pair_or_path=qa_train["context"],
                         val_texts_or_path=qa_en["validation"]["question"][-eval_examples:],
                         val_text_pair_or_path=qa_en["validation"]["context"][-eval_examples:],
                         labels_or_path=[a["text"][0] for a in qa_train["answers"]],
                         val_labels_or_path=[a["text"][0] for a in qa_en["validation"]["answers"]][-eval_examples:],
                         train_question_categories=_get_en_squad_categories(qa_train),
                         val_question_categories=_get_en_squad_categories(qa_en["validation"])[-eval_examples:],
                         batch_size=1,
                         val_evaluators=val_metrics + superglue_metrics + concept_aware_metrics,
                         # val_evaluators=val_metrics,
                         source_lang_id="en",
                         objective_id="AQA-en")


# Teabreac dataset & objective:

def _get_answer(ans_object: List[Dict[str, str]]) -> str:
    # list([{'number': '121', 'date': {'day': '', 'month': '', 'year': ''}, 'spans': []}])
    return " ".join(ans['number'] if ans['number']
                    else " ".join(ans['date'].values()) if " ".join(ans['date'].values())
                    else " ".join(ans['spans']) for ans in ans_object)


def _get_categories(program_modules: List[str]) -> str:
    return " ".join(program_modules)


qa_train = pd.read_json("training/data/teabreac_v1.0_multihop_qa_train.jsonl", lines=True)
orig_len = len(qa_train)

qa_train = qa_train[qa_train["context_text"].apply(lambda text: len(text) < 1000)].sample(frac=1)
print("Reduced to %s percent of original samples by length." % (len(qa_train) / orig_len) * 100)

qa_train["context_text"] = qa_train["context_text"].apply(lambda c: c.replace(" -> ", ". "))
qa_train["answers_text"] = qa_train["answers_objects"].apply(lambda ans_obj: _get_answer(ans_obj))
qa_train["program_modules_str"] = qa_train["program_modules"].apply(lambda modules: _get_categories(modules))

qa_val = pd.read_json("training/data/teabreac_v1.0_multihop_qa_dev.jsonl", lines=True)
qa_val["context_text"] = qa_val["context_text"].apply(lambda c: c.replace(" -> ", ". "))
qa_val["answers_text"] = qa_val["answers_objects"].apply(lambda ans_obj: _get_answer(ans_obj))
qa_val["program_modules_str"] = qa_val["program_modules"].apply(lambda modules: _get_categories(modules))

qa_val = qa_val[qa_val["answers_text"].apply(lambda ans: ans is not None and isinstance(ans, str) and len(ans.strip()) > 0)]

teabreac = Priming(lang_module,
                   difficulty_sample=5,  # TODO set
                   demos_selection_strategy="hard",  # TODO set
                   texts_or_path=qa_train["question_text"],
                   text_pair_or_path=qa_train["context_text"],
                   val_texts_or_path=qa_val["question_text"][-eval_examples:],
                   val_text_pair_or_path=qa_val["context_text"][-eval_examples:],
                   labels_or_path=qa_train["answers_text"],
                   val_labels_or_path=qa_val["answers_text"][-eval_examples:],
                   train_question_categories=qa_train["program_modules_str"],
                   val_question_categories=qa_val["program_modules_str"][-eval_examples:],
                   batch_size=1,
                   val_evaluators=val_metrics,
                   # val_evaluators=val_metrics,
                   source_lang_id="en",
                   objective_id="AQA-en")

schedule = ParallelSchedule(objectives=[q_answering_en, teabreac],
                            args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()
