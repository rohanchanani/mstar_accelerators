"""Platform-owned model corpus. No acceleration names or implementation recipes.

The small pinned descriptions ship with the harness so registry matching never
requires downloading weights or guessing a checkpoint from a node class.
"""
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ModelCase:
    key: str
    model_name: str

    @property
    def directory(self):
        return Path(__file__).with_name('corpus_data') / self.key

    @property
    def manifest(self):
        return json.loads((self.directory / 'assets.json').read_text())

    def describe(self, checkpoint=None):
        from mstar.model.registry import get_model_class, model_init_kwargs
        from .assets import verify
        if checkpoint is None:
            for entry in self.manifest['files']:
                path = self.directory / entry['path']
                if path.exists():
                    verify(path, entry)
        return get_model_class(self.model_name).describe(**(
            model_init_kwargs(self.model_name) | {
                'model_path_hf': str(checkpoint or self.directory)}))

    def identity(self):
        manifest = self.manifest
        return {'corpus': self.key, 'model': self.model_name,
                'checkpoint': manifest['model'], 'revision': manifest['revision']}


# New logical semantics/configurations extend this central corpus and the
# model-level case constructors. A new physical implementation never does.
MODELS = (ModelCase('whisper_tiny', 'whisper'), ModelCase('qwen3_omni', 'qwen3_omni'))


def logical_nodes(model):
    """Use the real graph and fresh logical shells, without physical programs."""
    walks = {}
    for walk, graph in model.get_graph_walk_graphs().items():
        for name in graph.get_nodes():
            walks.setdefault(name, []).append(walk)
    nodes, unavailable = {}, {}
    for name, hosted in walks.items():
        try:
            shell = model.create_submodule(name)
        except NotImplementedError as exc:
            unavailable[name] = str(exc)
            continue
        if shell is not None:
            nodes[name] = (shell, tuple(hosted))
    return nodes, unavailable
