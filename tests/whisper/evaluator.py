"""Host evaluator of the builder's ggml operations, solely for interface tests.

This is immediate Torch execution, not a deployment IR or a replacement
Whisper architecture.  The independent oracle is the original model program.
It cannot establish Vulkan shader or hardware correctness.
"""
import torch
import torch.nn.functional as F

from backends.pi.torch.tests.categorical_reference import categorical_row


class EagerGraph:
    def __init__(self, bindings):
        self.bindings = bindings
        self.outputs = {}

    def tensor(self, name, ne, dtype="f32"):
        value = self.bindings[name]
        ne = tuple(ne) + (1,) * (4 - len(ne))
        expected = {"f32": torch.float32, "i32": torch.int32, "i64": torch.int64}[dtype]
        assert value.dtype == expected, (name, value.dtype, expected)
        assert value.numel() == torch.tensor(ne).prod(), (name, value.shape, ne)
        return value.view(tuple(reversed(ne)))

    def mul_mat(self, a, b):
        return b @ a.transpose(-1, -2)

    def add(self, a, b): return a + b
    def mul(self, a, b): return a * b
    def scale(self, a, scale): return a * scale
    def norm(self, a, eps): return F.layer_norm(a, (a.shape[-1],), eps=eps)
    def gelu_erf(self, a): return F.gelu(a, approximate="none")

    def softmax(self, a, mask=None, scale=1.):
        a = a * scale
        return (a if mask is None else a + mask).softmax(dim=-1)

    def reshape(self, a, ne):
        ne = tuple(ne) + (1,) * (4 - len(ne))
        return a.reshape(tuple(reversed(ne)))

    def permute(self, a, axes):
        return a.permute(tuple(3 - axes.index(i) for i in reversed(range(4))))

    def cont(self, a): return a.contiguous()

    def view(self, a, ne, strides_bytes, offset=0):
        ne = tuple(ne) + (1,) * (4 - len(ne))
        strides = list(strides_bytes)
        while len(strides) < 4:
            strides.append(strides[-1] * ne[len(strides) - 1])
        return torch.as_strided(a, tuple(reversed(ne)), tuple(s // a.element_size() for s in reversed(strides)),
                                a.storage_offset() + offset // a.element_size())

    def get_rows(self, a, indices):
        assert a.shape[:2] == (1, 1)
        return a[..., indices.reshape(-1).long(), :]

    def set_rows(self, a, indices, dst):
        dst[..., indices.reshape(-1).long(), :] = a
        return dst

    def im2col(self, kernel, x, stride=1, padding=0, dilation=1):
        channels, length = x.shape[-2:]
        columns = F.unfold(x.reshape(1, channels, 1, length),
                           kernel_size=(1, kernel.shape[-1]), dilation=(1, dilation),
                           padding=(0, padding), stride=(1, stride))
        return columns.transpose(1, 2).unsqueeze(0).contiguous()

    def argmax(self, a):
        # ggml reduces dim0; its result's ne0 becomes the previous ne1.
        return a.argmax(dim=-1).to(torch.int32).unsqueeze(0)

    def categorical(self, logits, seeds, offsets, temperatures):
        # This explicit test interpreter is never a serving implementation.
        # GPU arithmetic and dispatch still require native target conformance.
        rows = logits.reshape(-1, logits.shape[-1])
        tokens = [categorical_row(row.tolist(), int(seed), int(offset), float(temperature))
                  for row, seed, offset, temperature in
                  zip(rows, seeds.flatten(), offsets.flatten(), temperatures.flatten(), strict=True)]
        return torch.tensor(tokens, dtype=torch.int64).reshape(1, 1, 1, -1)

    def cast(self, a, dtype):
        return a.to({"f32": torch.float32, "i32": torch.int32, "i64": torch.int64}[dtype])

    def output(self, name, a):
        self.outputs[name] = a.clone()

    def effect(self, a): pass


class ReadOnlyAlgebraGraph(EagerGraph):
    """FP64 algebra of exact logical bindings, currently used by the encoder.

    Validate the graph's declared storage types before widening their values.
    This is not a physical FP32 execution or a rounding emulator. Byte views,
    representation casts and state writes require semantics this consumer does
    not provide; the ordinary EagerGraph retains those integration-test duties.
    """

    def tensor(self, name, ne, dtype="f32"):
        value = super().tensor(name, ne, dtype)
        return value.to(torch.float64) if value.is_floating_point() else value

    def view(self, a, ne, strides_bytes, offset=0):
        raise NotImplementedError("algebra evaluation has no physical byte views")

    def set_rows(self, a, indices, dst):
        raise NotImplementedError("algebra evaluation has no mutable state")

    def cast(self, a, dtype):
        raise NotImplementedError("algebra evaluation has no representation casts")

    def effect(self, a):
        raise NotImplementedError("algebra evaluation has no state effects")
