from typing import List

from adaptor.adapter import Adapter
from adaptor.evaluators.generative import ROUGE
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset

from priming_objective import Priming
from training.all_evaluators import info_demos_evaluators, random_demos_evaluators, eval_examples
from training.teabreac_evaluators import tea_train, tea_val, per_concepts_eval_objective, hard_concepts, mean_concepts, \
    easy_concepts, tea_train_subset
from training.all_evaluators import superglue_evaluators

training_arguments = AdaptationArguments(output_dir="train_dir_teabreac+qa_info_large",
                                         learning_rate=2e-5,  # we set LR=2e-4 for pre-training experiments
                                         # stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=300000,
                                         gradient_accumulation_steps=30,  # TODO: set
                                         eval_steps=200,  # TODO: set
                                         logging_steps=10,
                                         save_steps=1000,
                                         num_train_epochs=2,
                                         evaluation_strategy="steps",
                                         save_total_limit=10,
                                         stopping_patience=30)


def _construct_priming_prompt(previous_examples: List[str], current_example: str) -> str:
    return " ".join(previous_examples + [current_example])


# lang_module = LangModule("google/mt5-small")  # TODO set
# lang_module = LangModule("gaussalgo/mt5-base-priming-QA_en-cs")
# lang_module = LangModule("google/mt5-base")
# lang_module = LangModule("Helsinki-NLP/opus-mt-en-cs")
lang_module = LangModule("google/mt5-large")

val_metrics = [ROUGE(**{"additional_sep_char": "▁"})]

# Adversarial QA dataset & objective:
qa_en = load_dataset("adversarial_qa", "adversarialQA")
qa_train = qa_en["train"].filter(lambda entry: len(entry["context"]) < 2000)


def _get_en_qa_categories(data) -> List[str]:
    return [question.split()[0] if not question.startswith("To")
            else " ".join(question.split()[:2])
            for question in data["question"]]


qa_objective = Priming(lang_module,
                       max_eval_samples=eval_examples,
                       # difficulty_sample=5,  # TODO set
                       demos_selection_strategy="informative",  # TODO set
                       texts_or_path=qa_train["question"],
                       text_pair_or_path=qa_train["context"],
                       val_texts_or_path=qa_en["validation"]["question"],
                       val_text_pair_or_path=qa_en["validation"]["context"],
                       labels_or_path=[a["text"][0] for a in qa_train["answers"]],
                       val_labels_or_path=[a["text"][0] for a in qa_en["validation"]["answers"]],
                       train_question_categories=_get_en_qa_categories(qa_train),
                       val_question_categories=_get_en_qa_categories(qa_en["validation"]),
                       batch_size=1,
                       val_evaluators=val_metrics + superglue_evaluators + info_demos_evaluators + random_demos_evaluators,  # TODO: test reusing cache
                       source_lang_id="en",
                       objective_id="AQA-en")

# Teabreac dataset & objective:

teabreac_train = Priming(lang_module,
                         max_eval_samples=eval_examples,
                         # difficulty_sample=5,  # TODO set
                         demos_selection_strategy="informative",  # TODO set
                         texts_or_path=tea_train_subset["question_text"],
                         text_pair_or_path=tea_train_subset["context_text"],
                         val_texts_or_path=tea_val["question_text"],
                         val_text_pair_or_path=tea_val["context_text"],
                         labels_or_path=tea_train_subset["answers_text"],
                         val_labels_or_path=tea_val["answers_text"],
                         train_question_categories=tea_train_subset["program_modules_str"],
                         val_question_categories=tea_val["program_modules_str"],
                         batch_size=1,
                         val_evaluators=val_metrics,
                         source_lang_id="en",
                         objective_id="teabreac_train-en")

teabreac_per_concept_evals = [per_concepts_eval_objective(lang_module, tea_train, easy_concepts, label="easy"),
                              per_concepts_eval_objective(lang_module, tea_train, mean_concepts, label="mean"),
                              per_concepts_eval_objective(lang_module, tea_train, hard_concepts, label="hard")]

schedule = ParallelSchedule(objectives=[qa_objective, teabreac_train],
                            extra_eval_objectives=teabreac_per_concept_evals,
                            args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()
