import json
from typing import List

from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset

from evaluation.tasks.en.superglue import all_task_classes
from priming_objective import Priming
from training.sglue_evaluators import TaskROUGE

training_arguments = AdaptationArguments(output_dir="train_dir",
                                         learning_rate=5e-5,  # we set LR=2e-4 for pre-training experiments
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
                                         num_train_epochs=50,
                                         evaluation_strategy="steps",
                                         save_total_limit=10,
                                         stopping_patience=30)
eval_examples = 200  # TODO set

# priming
num_demonstrations = 3

val_metrics = [BLEU(**{"additional_sep_char": "▁"}, decides_convergence=True)]

superglue_metrics = [TaskROUGE(TaskCls(), num_demonstrations, firstn=eval_examples // 3) for TaskCls in
                     all_task_classes]


def _construct_priming_prompt(previous_examples: List[str], current_example: str) -> str:
    return " ".join(previous_examples + [current_example])


# lang_module = LangModule("google/mt5-small")  # TODO set
# lang_module = LangModule("gaussalgo/mt5-base-priming-QA_en-cs")  # TODO set
lang_module = LangModule("google/mt5-base")
# lang_module = LangModule("google/mt5-large")

# priming
per_type_examples = {}

qa_en = load_dataset("adversarial_qa", "adversarialQA")
qa_train = qa_en["train"].filter(lambda entry: len(entry["context"]) < 2000)


def _get_en_squad_categories(data) -> List[str]:
    return [question.split()[0] if not question.startswith("To")
            else " ".join(question.split()[:2])
            for question in data["question"]]


q_answering_en = Priming(lang_module,
                         difficulty_sample=64,  # TODO set
                         demos_selection_strategy="random",  # TODO set
                         texts_or_path=qa_train["question"],
                         text_pair_or_path=qa_train["context"],
                         val_texts_or_path=qa_en["validation"]["question"][-eval_examples:],
                         val_text_pair_or_path=qa_en["validation"]["context"][-eval_examples:],
                         labels_or_path=[a["text"][0] for a in qa_train["answers"]],
                         val_labels_or_path=[a["text"][0] for a in qa_en["validation"]["answers"]][-eval_examples:],
                         train_question_categories=_get_en_squad_categories(qa_train),
                         val_question_categories=_get_en_squad_categories(qa_en["validation"])[-eval_examples:],
                         batch_size=1,
                         val_evaluators=val_metrics + superglue_metrics,
                         # val_evaluators=val_metrics,
                         source_lang_id="en",
                         objective_id="AQA-en")

squad_cs_dataset = json.load(open("training/data/czech_squad_10-sents-abs.json"))

skipped = 0

questions_cs = []
contexts_cs = []
answers_cs = []
categories_cs = []

for i, entry in squad_cs_dataset.items():
    if len(entry["context"]) > 8000:
        skipped += 1
        continue

    questions_cs.append(entry["question"])
    contexts_cs.append(entry["context"])
    answers_cs.append(entry["answers"]["text"][0])
    categories_cs.append(entry["answer_type"])

print("Skipped cs examples: %s" % skipped)

q_answering_cs = Priming(lang_module,
                         difficulty_sample=64,  # TODO set
                         demos_selection_strategy="random",  # TODO set
                         texts_or_path=questions_cs[:-eval_examples],
                         text_pair_or_path=contexts_cs[:-eval_examples],
                         val_texts_or_path=questions_cs[-eval_examples:],
                         val_text_pair_or_path=contexts_cs[-eval_examples:],
                         labels_or_path=answers_cs[:-eval_examples],
                         val_labels_or_path=answers_cs[-eval_examples:],
                         train_question_categories=categories_cs[:-eval_examples],
                         val_question_categories=categories_cs[-eval_examples:],
                         batch_size=1,
                         val_evaluators=val_metrics,
                         source_lang_id="cs",
                         objective_id="SQUAD-cs")

schedule = ParallelSchedule(objectives=[q_answering_en, q_answering_cs],
                            args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()
