"""Task registry. Every backend consumes checked plans and returns artifacts/metadata."""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class TaskSpec:
    name: str
    backend: str
    planner: str


TASK_SPECS = {
    'mdur': TaskSpec('mdur', 'browser', 'mdur_plan'),
    'miqa': TaskSpec('miqa', 'raster', 'miqa_plan'),
    'msocr': TaskSpec('msocr', 'raster', 'msocr_plan'),
    'mgui': TaskSpec('mgui', 'browser', 'mgui_plan'),
}


def get_task(name: str):
    if name not in TASK_SPECS:
        raise ValueError(f'Unknown rendering task: {name}')
    spec = TASK_SPECS[name]
    module = import_module(f'{__name__}.{name}')
    return spec, module, getattr(module, spec.planner)
