from typing import List

from adaptor.adapter import Adapter
from adaptor.evaluators.generative import ROUGE
from adaptor.lang_module import LangModule
from adaptor.schedules import SequentialSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset

from training.fewshot_objective import InContextFewShot
from training.teabreac_evaluators import tea_val, tea_train_subset

training_arguments = AdaptationArguments(output_dir="train_dir_teabreac+qa_info_large",
                                         learning_rate=2e-5,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=300000,
                                         gradient_accumulation_steps=30,
                                         eval_steps=1000,
                                         logging_steps=50,
                                         save_steps=1000,
                                         num_train_epochs=5,
                                         evaluation_strategy="steps",
                                         save_total_limit=10,
                                         stopping_patience=30)


def _construct_priming_prompt(previous_examples: List[str], current_example: str) -> str:
    return " ".join(previous_examples + [current_example])


lang_module = LangModule("google/t5-large-lm-adapt")

val_metrics = [ROUGE(**{"additional_sep_char": "▁"})]

# Adversarial QA dataset & objective:
qa_en = load_dataset("adversarial_qa", "adversarialQA")
qa_train = qa_en["train"].filter(lambda entry: len(entry["context"]) < 2000)


def _get_en_qa_categories(data) -> List[str]:
    return [question.split()[0] if not question.startswith("To")
            else " ".join(question.split()[:2])
            for question in data["question"]]


eval_examples = 200


qa_objective = InContextFewShot(lang_module,
                                max_eval_samples=eval_examples,
                                difficulty_sample=20,
                                demos_selection_strategy="hard",
                                texts_or_path=qa_train["question"],
                                text_pair_or_path=qa_train["context"],
                                val_texts_or_path=qa_en["validation"]["question"],
                                val_text_pair_or_path=qa_en["validation"]["context"],
                                labels_or_path=[a["text"][0] for a in qa_train["answers"]],
                                val_labels_or_path=[a["text"][0] for a in qa_en["validation"]["answers"]],
                                train_question_categories=_get_en_qa_categories(qa_train),
                                val_question_categories=_get_en_qa_categories(qa_en["validation"]),
                                batch_size=1,
                                source_lang_id="en",
                                objective_id="AQA-en")

# Teabreac dataset & objective:

teabreac_train = InContextFewShot(lang_module,
                                  max_eval_samples=eval_examples,
                                  difficulty_sample=20,
                                  demos_selection_strategy="hard",
                                  texts_or_path=tea_train_subset["question_text"],
                                  text_pair_or_path=tea_train_subset["context_text"],
                                  val_texts_or_path=tea_val["question_text"],
                                  val_text_pair_or_path=tea_val["context_text"],
                                  labels_or_path=tea_train_subset["answers_text"],
                                  val_labels_or_path=tea_val["answers_text"],
                                  train_question_categories=tea_train_subset["program_modules_str"],
                                  val_question_categories=tea_val["program_modules_str"],
                                  batch_size=1,
                                  source_lang_id="en",
                                  objective_id="teabreac_train-en")

schedule = SequentialSchedule(objectives=[teabreac_train, qa_objective],
                              args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()
