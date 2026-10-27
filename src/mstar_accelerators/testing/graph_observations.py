"""Observe actual graph sampling operands; no replacement decoder program.

Qualification adds copy-out roots to the graph passed to the normal native
provider. Computation and sampling operators are the original builder's. The
ordinary engine owns submission, completion and storage retention.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import importlib.util
from torch.utils._python_dispatch import TorchDispatchMode
import torch


class GraphSamplingObservation:
    def __init__(self):
        self.values = []

    @contextmanager
    def capture_torch(self, vocabulary):
        observer = self
        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if (func == torch.ops.aten.argmax.default and args
                        and args[0].ndim in (1, 2) and args[0].shape[-1] == vocabulary):
                    observer.values = [args[0].detach().clone()]
                return func(*args, **(kwargs or {}))
        with Mode():
            yield

    @contextmanager
    def install(self):
        from ..backends.graph import NativeGraphProvider
        original = NativeGraphProvider.prepare
        observer = self

        def prepare(provider, builder, constants, *, slots):
            shapes = {}

            class ObservedGraph:
                def __init__(self, graph):
                    self.graph, self.index = graph, 0
                def __getattr__(self, name):
                    return getattr(self.graph, name)
                def record(self, logits):
                    name = f"__qualification_sample_{self.index}"
                    self.index += 1
                    shapes[name] = tuple(reversed(logits.ne))
                    self.graph.output(name, logits)
                def argmax(self, logits):
                    self.record(logits)
                    return self.graph.argmax(logits)
                def categorical(self, logits, seeds, offsets, temperatures):
                    self.record(logits)
                    return self.graph.categorical(logits, seeds, offsets, temperatures)

            def observed_builder():
                from mstar_pi.ggml import Graph
                graph = Graph(max_nodes=4096)
                result = builder(graph=ObservedGraph(graph))
                return result.graph if isinstance(result, ObservedGraph) else result
            program = original(provider, observed_builder, constants, slots=slots)

            class Lease:
                def __init__(self, lease): self.lease = lease
                def submit(self, inputs, outputs):
                    device = next(iter(outputs.values())).device
                    saved = {key: torch.empty(shape, dtype=torch.float32, device=device)
                             for key, shape in shapes.items()}
                    self.lease.submit(inputs, outputs | saved)
                    observer.values = list(saved.values())
                def close(self): self.lease.close()
            class Program:
                def try_acquire(self):
                    lease = program.try_acquire()
                    return None if lease is None else Lease(lease)
                def close(self): program.close()
            return Program()
        with ExitStack() as stack:
            stack.enter_context(patch.object(NativeGraphProvider, 'prepare', prepare))
            # The native CPU decoder already returns its actual readout logits.
            # Observe that backend boundary without replacing any arithmetic.
            module = 'mstar_accelerators.implementations.whisper.cpu_ggml_decoder'
            if importlib.util.find_spec(module) is not None:
                from importlib import import_module
                decoder = import_module(module).Decoder
                original_run = decoder.run
                def run(owner, call):
                    result = original_run(owner, call)
                    observer.values = [result[0].detach().clone()]
                    return result
                stack.enter_context(patch.object(decoder, 'run', run))
            yield self
