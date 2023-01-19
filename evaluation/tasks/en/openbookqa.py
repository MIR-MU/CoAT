from typing import Dict, Any, Optional
import spacy

from datasets import load_dataset, concatenate_datasets
from promptsource.templates import Template, DatasetTemplates

from evaluation.tasks.task import Task
from training.fewshot_objective import priming_formats


class OpenBookQATask(Task):

    def _construct_qa_prompt(self, question: str, context: str) -> str:
        return priming_formats["QA"][self.lang_id] % (question, context)

    def _informative_factor(self, sample: Dict[str, Any]) -> str:
        return next(str(token).lower() for token in self.spacy(sample["fact1"]) if token.dep_ == "ROOT")

    def __init__(self,
                 lang_id: str,
                 template: Optional[Template] = DatasetTemplates("openbookqa/additional")["which_correct"]):
        super().__init__()
        self.lang_id = lang_id
        self.spacy = spacy.load('en_core_web_sm')
        self.template = template.name

        dataset = concatenate_datasets([load_dataset("openbookqa", "additional")["validation"],
                                        load_dataset("openbookqa", "additional")["test"]])

        self.data = [(*template.apply(sample),  # type: ignore
                      self._informative_factor(sample)) for sample in dataset]
