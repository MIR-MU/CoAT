from typing import Optional

from datasets import load_dataset
from promptsource.templates import Template, DatasetTemplates

from evaluation.tasks.task import Task


class GLUEDiagnostics(Task):

    def __init__(self,
                 lang_id: str = "en",
                 template: Optional[Template] = DatasetTemplates('glue/mnli')["GPT-3 style"]):
        super().__init__(".")
        self.lang_id = lang_id
        self.seen_premises = set()
        self.template = template.name

        dataset = load_dataset("pietrolesci/glue_diagnostics")["test"]

        # this will remove the samples with the identical premise from the evaluation
        duplicates_map = []
        for sample in dataset:
            identifier = sample["logic"] + " ".join(sorted(sample["premise"].split()))
            duplicates_map.append(identifier in self.seen_premises)
            self.seen_premises.add(identifier)

        self.data = [(*template.apply(sample), sample["logic"]) for sample, is_duplicate  # type: ignore
                     in zip(dataset, duplicates_map) if sample["logic"] and not is_duplicate]
