# V3D CSD completion contract, private Pi fork v1

**Status: this is the unchanged Stage 1 kernel contract and repair. The target investigator built the complete matching kernel/modules/DTBs and booted the confirmed `6.18.50-mstar-pi-stage1-r2` candidate; Stage 1 and Stage 2 normal-path device gates passed on it. Kernel fault injection, cancellation/reset, failed-child-fence and KASAN lifetime qualification remain open. Do not restage this kernel merely to build the new userspace candidate. The historical source analysis below is not a production exceptional-path certification.**

This patch closes specific source-established holes underneath the native completion API. It does not replace the DRM scheduler, implement GPU recovery, prove a QPU program correct, or claim that any error-signaled fence proves that physical writes have stopped.

## 1. Source and supported scope

The patch targets the Raspberry Pi source package **`1:6.18.50-1+rpt1`**, after the relevant distribution patches, as supplied in the r2 packet. The reported running kernel was `6.18.50+rpt-rpi-2712`. `source_manifest.json` pins the before and after SHA-256 of every changed file. Apply only when all before hashes match; do not fuzzy-apply it to another kernel and treat it as the same repair.

The packet contains complete selected V3D/DRM scheduler sources, not a complete kernel build tree. We read the selected files and their scheduler consumers; we did not rebuild the kernel here. Packet archive provenance records:

- `linux_6.18.50.orig.tar.xz`: `d2fc041dab4e11d9645e3ba53be058faa52b8ce28a8a97c889bb2cacee170461`
- `linux_6.18.50-1+rpt1.debian.tar.xz`: `98c2e147fda1eac197a221e6fb824e794c04830112d306f32c836433833e355e`
- Source URLs are under `https://archive.raspberrypi.com/debian/pool/main/l/linux/`.

The packet says its source package signature was **not verified** and its binary was **not rebuilt**. Codex should obtain the full package through the configured signed Raspberry Pi package metadata and record the verification result. Matching selected hashes establish patch compatibility, not provenance of the running binary.

The primary repaired path is direct `DRM_IOCTL_V3D_SUBMIT_CSD`, used by the native backend. The same late dependency allocation is repaired for CPU-indirect CSD because it feeds the same scheduler and cleanup queue. **This is not an audit of every CPU-extension parser, CL/TFU error path, implicit-synchronization rule, or arbitrary external provider.** A provider using additional paths must qualify those paths before it is enabled. In particular, the unchanged CL submit path has its own late dependency allocations; v1 is not a blanket no-partial-submission promise for CL.

Inputs and output sync objects must be valid and owned across the ioctl. The native runtime must continue supplying explicit dependencies, retaining all referenced BOs, checking its output fence, and treating any unknown acceptance or negative completion as fatal. The patch does not make invalid or concurrently destroyed sync handles safe, validate shader addresses, or infer omitted dependencies.

## 2. Changes, by obligation

### Acceptance is not followed by an allocation that can reject the invocation

Upstream `v3d_submit_csd_ioctl()` pushes CSD before calling the fallible `drm_sched_job_add_dependency()` for the cache-clean job. The clean owns the invocation's BO references. If that allocation fails, returning an ioctl error and destroying the clean does not undo the accepted CSD.

The patch reserves one ordinary scheduler dependency slot **before** the first arm/push. After the parent is armed, it replaces that slot's stub fence with the parent's finished fence. Replacing an existing xarray entry does not require allocation. There is no error-returning branch after the first push on the repaired direct CSD path. CPU-indirect CSD reserves both links first.

This deliberately uses the scheduler's normal dependency array. `drm_sched_entity_kill_jobs_work()` waits those dependencies before killing queued jobs; it does not consume arbitrary `prepare_job` dependencies. We therefore do not introduce a special dependency side channel. `drm_sched_job_arm()` remains the scheduler's documented point of no return.

Source anchors in the packet: `v3d_submit.c`, `v3d_submit_csd_ioctl` (original line 1199), `v3d_submit_cpu_ioctl` (1309), and `v3d_setup_csd_jobs_and_bos` (297); `sched_main.c`, `drm_sched_job_arm` (840) and `drm_sched_job_add_dependency`; `sched_entity.c`, `drm_sched_entity_kill_jobs_work` and `drm_sched_job_dependency`. The [Linux xarray documentation](https://docs.kernel.org/core-api/xarray.html) explains preallocation/reserved entries and replacement semantics.

### Completion reports known failures

- A CSD hardware-fence allocation error is returned as `ERR_PTR(error)`, not `NULL` (which the DRM scheduler treats as synchronous success).
- CSD, CPU, and clean execution inspect error status of their dependencies. Ordering a dependency does not itself propagate that dependency's error.
- Cache-clean timeouts now return `-ETIMEDOUT` from all three branches. The clean job propagates that error, including a device/reset failure observed during cleaning. The power-suspend caller also handles the new return value instead of treating a failed clean as permission to power down.
- Observed MMU/GMP fault IRQs latch a fatal device error. A later IRQ completion propagates an already-known fatal error to its hardware fence. Userspace must still check device health/reset epoch around completion: an error can be observed after an individual fence was signaled, and already-signaled fences cannot be retroactively rewritten.

This is bounded failure reporting, not a claim that every possible hardware fault is detected by the driver. An instruction sequence that computes an incorrect value without a hardware fault remains an implementation-correctness problem.

Source anchors: `v3d_sched.c`, `v3d_csd_job_run` (367), `v3d_cache_clean_job_run` (706), and `v3d_cpu_job_run`; `v3d_gem.c`, `v3d_clean_caches` (214); `v3d_irq.c`, `v3d_irq` / `v3d_hub_irq`; `sched_main.c`, `drm_sched_run_job_work`.

### Failure does not authorize physical reuse

A failed or cancelled cache-clean job does not establish that CSD writeback has settled. The patched scheduler retains an extra reference on that job. Its BO owners, GEM mappings, PM reference, and associated driver owners remain alive after file close. It similarly retains affected jobs when the reset epoch changes or the device is poisoned. A successful clean is marked explicitly so ordinary retirement remains unchanged.

Each initialized job holds module, DRM-device, platform-device, and per-file private-state references. The private file state includes statistics and the perfmon lookup table: CPU performance queries can still look up that table while jobs finish. The scheduler entities close at `postclose`, but private/perfmon state is freed only after its last job reference drops. This also protects late IRQ/timeout bookkeeping.

The first fatal failure acquires a permanent device/module hold. The device then rejects CL, TFU, CSD, and CPU submissions at entry and immediately before acceptance. Queued executions check the fatal state before starting. An invocation racing the fatal transition may already have been accepted; it still has an ordinary final fence and its owners are retained. The check is not a promise to atomically cancel running hardware.

**The retention has no release or recovery operation in v1. Controlled reboot is the recovery policy.** It intentionally leaks the affected owners rather than trusting an error fence, a timeout, process exit, or a reset attempt as evidence of safe reuse. Repeated work is rejected so the leak is bounded by the work accepted when failure occurred. User-space quarantine is still required to prevent logical reuse of retained allocations; kernel quarantine protects against deallocation/reassignment when the client dies.

Normal module unload is prevented by retained module references after poison. The platform driver's bind/unbind attributes are suppressed. A `get_device()` reference alone does not prevent driver removal: **forced module removal, platform-device deletion/DT-overlay hot removal, manual unbind mechanisms outside these attributes, and in-place kernel recovery are outside v1**. Use a dedicated, fixed Pi SoC device and reboot. Do not describe this as a hot-unplug-safe backend.

### Reset cannot silently replay stateful CSD

The existing reset sequence remains, but enters fatal state before stopping/resetting schedulers. CSD jobs also carry their submission reset epoch. A resubmission callback from another epoch returns failure before touching the submission registers. Other queue callbacks refuse execution after fatal state. Reset counters remain diagnostic; there is no userspace acknowledgement that clears poison or authorizes replay.

`v3d_reset()` itself is not changed into a proof of quiescence. Its existing helper failures are another reason to retain ownership until reboot. Cancellation can conservatively poison the device when it skips a needed clean even if a particular job ultimately had no outstanding writes. That availability cost is intentional for this first version.

Source anchors: `v3d_sched.c`, `v3d_gpu_reset_for_timeout` (722), `v3d_sched_job_free`; `v3d_drv.c`, `v3d_postclose` (179); `v3d_submit.c`, `v3d_job_init` / `v3d_job_free`; `v3d_perfmon.c`, `v3d_perfmon_find` / `v3d_perfmon_close_file`.

## 3. Explicit private ABI gate

These are **private fork IDs**, not upstream allocations. No additional ioctl is introduced.

| `DRM_IOCTL_V3D_GET_PARAM` parameter | Result |
| --- | --- |
| `0x50490001` — `DRM_V3D_PARAM_CSD_COMPLETION_CONTRACT` | Exactly `1` for this contract on CSD hardware; `0` on unsupported hardware. |
| `0x50490002` — `DRM_V3D_PARAM_CSD_DEVICE_STATUS` | `0` healthy; positive errno for the first sticky fatal failure. |

The native backend must require version **exactly 1**, healthy status, and a stable global reset counter. `EINVAL`, a missing query, a different version, or a nonzero status is rejection. Do not add a kernel-version-string allowlist, environment override, or “trust this driver” file.

The capability identifies code behavior. It does not certify that this candidate patch was built correctly or qualified on this Pi. Preserve the source/config/binary hash chain in the evidence.

The read-only probe performs no allocation or GPU submission:

```sh
python3 kernel/tests/query_contract.py /dev/dri/renderD128
```

It requires a little-endian aarch64 Pi and verifies the sysfs driver is `v3d`. On stock r2 it should reject with exit 4. On a healthy patched driver it prints contract `1`, fatal errno `0`, and reset counters. Nonzero fatal status exits 5. Select the render node from the actual sysfs inventory; do not assume `renderD128` blindly.

## 4. What was run here

```sh
python3 kernel/tests/check_source.py \
  --source-root /absolute/path/to/r2/sources/kernel-package \
  --report /absolute/path/to/new-results/kernel-source-validation.json
```

`source_validation.json` records the result: eight exact preimage/postimage matches, twenty structure checks, and twenty-seven branch cases compiled and run with address/undefined-behavior sanitizers. LeakSanitizer is disabled because this sandbox does not permit its `/proc` thread inspection. The harness itself allocates no dynamic kernel owners.

The tests extract the **actual patched function bodies**, then supply controlled substitutes for scheduler/xarray/device/register boundaries. They exercise pre-acceptance dependency allocation failure, accepted CSD/clean publication, fence-allocation failure, dependency error propagation, cache-clean timeout branches, skipped-clean retirement, reset rejection without submission writes, and sticky ownership retention.

**These are not full kernel tests.** They do not test the real xarray allocator, scheduler concurrency, hard IRQ timing, module removal, MMU behavior, or physical cache writeback. A pass must not be relabeled as “kernel compiled” or “exception drain proven on hardware.”

## 5. Codex: build and boot gates

1. Preserve the current working kernel, modules, boot configuration, and r2 evidence. Create a separate source/build/staging directory. Never overwrite the running driver's files with an out-of-tree object from a different configuration.
2. Obtain the complete matching Raspberry Pi source package and apply its normal packaging patches. `apt-get source linux=1:6.18.50-1+rpt1` is one route when that exact source version and `deb-src` metadata are available. Record the actual source verification and patch series. The packet's selected files already include their relevant packaging patches; do not apply the packet's selected rpi diff a second time to an already prepared tree.
3. Run `check_source.py` against that prepared tree. It checks without modifying it. A before-hash mismatch is a stop condition, not permission to use `--fuzz`.
4. Apply the patch in the dedicated tree with `patch --batch --fuzz=0 -p1`. Run the kernel's `scripts/checkpatch.pl --no-tree` on the patch and inspect warnings. Record any source/API correction as a new reviewed patch and regenerate hashes/tests.
5. Build the **complete kernel** with the actual Pi configuration and a unique local version. A representative native arm64 sequence is below. Resolve the target's actual config and boot packaging rather than guessing them.

```sh
# Run in the dedicated, prepared full kernel source tree.
# PI_BUILD and PI_STAGE are new absolute directories, never system paths.
mkdir -p "$PI_BUILD" "$PI_STAGE"
cp "/boot/config-$(uname -r)" "$PI_BUILD/.config"
scripts/config --file "$PI_BUILD/.config" \
  --set-str LOCALVERSION '-mstar-pi-v3d1' --disable LOCALVERSION_AUTO
make O="$PI_BUILD" ARCH=arm64 olddefconfig
make O="$PI_BUILD" ARCH=arm64 -j2 Image modules dtbs
make O="$PI_BUILD" ARCH=arm64 modules_install INSTALL_MOD_PATH="$PI_STAGE"
make O="$PI_BUILD" ARCH=arm64 kernelrelease
```

If `/boot/config-$(uname -r)` is absent, obtain the actual running config from the existing package or `/proc/config.gz`; do not silently substitute a generic config. Verify `CONFIG_DRM_V3D` is enabled and the resulting kernel release differs from the preserved release. Choose parallelism to fit the available RAM. For cross compilation, supply the verified cross toolchain; record it.

6. Package/stage the matching image, DTBs, overlays, modules, config, and hashes. Inspect this Pi's real boot configuration and establish the recoverable old-kernel selection before changing it. Do not use a blind `make install` or replace the only bootable image. This packet does not know the user's current boot-storage setup; Codex must discover it in the real environment.
7. Boot the candidate through that recoverable path. Record `uname -a`, the loaded module identity/hash, boot log, V3D probe log, and the read-only capability query. **Do not start the native/Torch GPU tests unless the explicit query succeeds.** A normal boot does not satisfy the remaining failure-path gate.

## 6. Validation on the candidate kernel

### First normal-path gate

Use the native tests in the parent package to submit deterministic tiny CSD programs with exact output/guard checks, ordered dependencies, two-prerequisite joins, independent contexts, and event snapshots. Test normal close after drain and a spawned worker. Record both fence status and device health/reset epoch; a signaled fence alone is insufficient.

Then rerun the r2 direct-QPU → Vulkan → direct-QPU shared-BO chain under the candidate kernel. Qualify the actual Vulkan provider path used, not the existence of a Vulkan device. Inspect its ioctl trace so that an unchanged CL/TFU/CPU-extension path is not silently brought into the v1 claim. No physical-overlap or performance claim follows from correct asynchronous ordering.

### Focused kernel failure gate

Before describing this as a production failure contract, use a debug/test kernel and reviewable, temporary fault injection to test these exact boundaries. Keep those injection hooks out of the shipped patch. Prefer a kernel test harness for software branches; do not induce an uncontrolled GPU hang as a first experiment.

- Force the reserved dependency allocation to fail before the first push. Confirm negative ioctl return, zero submitted CSD jobs, no changed output fence, and ordinary owner release.
- Force hardware-fence creation to return `-ENOMEM`. Confirm the final clean fence reports failure, device health becomes fatal, and no CSD register kick occurred.
- Force each cache-clean timeout branch separately. Confirm final error, kernel job/BO retention after closing the fd and killing the submitting process, and rejection from a new context. Do not infer that the forced error actually reproduced a stalled bus transaction.
- Exercise scheduler cancellation of a queued clean after its CSD dependency. Confirm the ordinary dependency drain occurs and cancelled clean ownership is retained.
- Exercise a reset epoch change without replaying the program. Confirm no second CSD kick, error/health reporting, and late IRQ handling without use-after-free.
- Exercise failure reporting from both prerequisites of a join. Verify the real combined fence carries child errors in this kernel; if the join's status hides a child failure, fix that producer/consumer boundary before enabling it.
- With KASAN/lockdep or equivalent diagnostics, cover process exit while CSD, clean, CPU query, and timeout bookkeeping are pending. Confirm the file-private/perfmon owner remains valid and ordinary successful jobs eventually release their refs. A quarantined owner must remain retained.

The physical failure experiments require an isolated Pi, a tested reboot/recovery path, and collection of kernel logs. Positive evidence should say exactly which injected/detected failure occurred. **No successful forced-error test proves all stalled-DMA/reset cases settled; v1 does not rely on settlement and does not permit reuse after them.**

Report any failure, source mismatch, unresolved lock/lifetime issue, or unavailable build dependency directly. Do not bypass the ABI gate, suppress a kernel warning, clear poison, or weaken the native ownership policy to finish the test.
