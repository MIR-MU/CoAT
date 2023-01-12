from evaluation.sensitivity_evaluator import RougeInformative, RougeRandom
from evaluation.tasks.en.adversarialqa import AdversarialQATask
from evaluation.tasks.en.glue_diagnostics import GLUEDiagnostics
from evaluation.tasks.en.openbookqa import OpenBookQATask
from evaluation.tasks.en.r4c_hotpotqa import R4CHotpotQATask
from evaluation.tasks.en.superglue import all_task_classes
from evaluation.tasks.en.worldtree_qa import WorldTreeQA
from training.sglue_evaluators import TaskROUGE

eval_examples = 200
sglue_examples = eval_examples // 3
num_demonstrations = 3

superglue_evaluators = [TaskROUGE(TaskCls(), num_demonstrations, firstn=sglue_examples) for TaskCls in all_task_classes]

qa_task = AdversarialQATask("en")

glue_task = GLUEDiagnostics("en")
hotpotqa = R4CHotpotQATask("en")
worldtree = WorldTreeQA("en")
openbook = OpenBookQATask("en")

info_demos_evaluators = [RougeInformative(qa_task, firstn=eval_examples),
                         RougeInformative(glue_task, firstn=eval_examples),
                         RougeInformative(hotpotqa, firstn=eval_examples),
                         RougeInformative(worldtree, firstn=eval_examples),
                         RougeInformative(openbook, firstn=eval_examples)]

random_demos_evaluators = [RougeRandom(qa_task, firstn=eval_examples, reuse_last_run=True),
                           RougeRandom(glue_task, firstn=eval_examples, reuse_last_run=True),
                           RougeRandom(hotpotqa, firstn=eval_examples, reuse_last_run=True),
                           RougeRandom(worldtree, firstn=eval_examples, reuse_last_run=True),
                           RougeRandom(openbook, firstn=eval_examples, reuse_last_run=True)]
