# mstar-pi: native execution beneath Torch

This backend owns Pi-device storage, asynchronous submission, stream dependencies,
stable events, checked completion and retirement. The Torch adapter exposes those
semantics through the existing `PrivateUse1` tensor/stream/event interfaces.
Native programs and the pinned ggml Vulkan provider consume canonical allocations
and join the same completion model. There is no CPU compute fallback.

Build and qualify the native extension against the retained source. Host
compilation and shader inspection do not establish target correctness or
performance. Native conformance, central acceleration qualification and ordinary
service checks cover distinct boundaries.

## Ownership

- `native/`: canonical DRM allocations, checked views, stream prefixes, event
  snapshots, provider admission, publication, health and retirement.
- `torch/`: Torch tensors and retained operators, staging, streams/events, native
  storage access and prepared native/ggml program bindings.
- `third_party/ggml/`: the pinned real ggml source and managed Vulkan integration.
  Graphs are actual ggml contexts and tensors. The provider imports runtime-owned
  allocations and never creates a separate weight or persistent-state heap.
- `kernel/`: the established driver status/completion contract and repair. A
  userspace rebuild is not a reason to restage the verified candidate kernel.
- `tools/run_hardware.py`: the shared physical-device lease and boot-scoped
  quarantine. Managed workers acknowledge native release explicitly.
- `tests/` and `torch/tests/`: source, stream, retained-consumer and provider gates.
  The existing Stage 1/Stage 2 fixtures and oracles remain regression requirements.

Model code and deployment decisions live in the surrounding acceleration library,
not in this backend. `mstar_pi.ggml.Graph` and `prepare_graph` have no Whisper name
cases. Their lease API supports capability-driven scheduling without a second
scheduler or a CPU-worker completion workaround.

## Contracts and graph interface

Read `docs/CONTRACT.md` for native/Torch ownership and `docs/PROVIDER.md` for the
submission/tail boundary. `docs/managed_graph_provider.md` describes complete
prepared graphs, canonical arenas, slot leases, supported operations and gates.
`docs/categorical.md` specifies the optional native categorical primitive,
immutable runtime seed/offset binding, exact RNG mapping and separate gates.
The C boundary is `native/include/mstar_pi/provider.h` and
`third_party/ggml/include/ggml-vulkan-managed.h`.

`mstar_pi.initialize()` is explicit and worker-local. Import and metadata graph
construction do not initialize a target. A `PreparedGraph` owns its prepared
contexts and canonical intermediate arenas; each `GraphLease` transfers to the
native runtime only upon submission. Immutable completion is independent of
slot reuse. `PreparedGraph.close()` and backend shutdown are ordered teardown,
never a serving-time substitute for asynchronous execution.

## Build outline

Use the accepted kernel contract and a matching Torch/compiler environment.
Build on the target with pinned dependencies and retain the build receipt:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
python -m unittest discover -s tests -v
cp build/shaders/transfer.spv torch/mstar_pi/transfer.spv
CC=gcc CXX=g++ MAX_JOBS=1 python -m pip install --no-build-isolation --no-deps ./torch
```

The root build compiles native code, the real ggml shaders and static libraries.
The native and ggml standard-library ABI must match official Torch 2.10.0. No
external installed ggml shared library or superseded Pi factory is used.

Before execution, record the actual loaded extension path and hash; an installed
package version alone does not identify its native operators:

```sh
python -c 'import hashlib, pathlib, mstar_pi._C as native; p = pathlib.Path(native.__file__); print(p, hashlib.sha256(p.read_bytes()).hexdigest())'
```

An explicitly selected preserved build must include its matching Python package,
not just a substituted shared object. Keep its source/build manifest with the
qualification receipt. Never silently use a stale venv extension when replaying
a newer GPU implementation.
