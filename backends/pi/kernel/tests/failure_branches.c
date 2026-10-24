/* SPDX-License-Identifier: MIT
 * Targeted execution of actual patched function bodies, extracted by
 * check_source.py. Kernel boundaries are substitutes, not a kernel emulator.
 * These tests cannot establish IRQ concurrency, GPU settlement, or ABI build.
 */
#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef uint32_t u32;
struct device { int refs; };
struct drm_device { struct device *dev; int refs; };
struct dma_fence { int status, refs, error; unsigned int seqno; };
struct v3d_fence { struct dma_fence base; unsigned int seqno; };
struct xarray { struct dma_fence *entries[16]; };
struct drm_sched_fence { struct dma_fence finished; };
struct drm_sched_job { struct xarray dependencies; struct drm_sched_fence *s_fence; };
struct kref { int count; };
struct v3d_job;
struct v3d_dev {
    struct drm_device drm;
    int csd_error, ver, cache_clean_lock, sched_lock;
    void *global_perfmon;
    unsigned int reset_counter;
    struct { struct v3d_job *active_job; } queue[4];
};
struct v3d_job {
    struct drm_sched_job base;
    struct v3d_dev *v3d;
    struct kref refcount;
    struct dma_fence *irq_fence, *done_fence;
    unsigned int reset_counter;
    bool needs_cache_clean, cache_clean_complete, quarantined;
    void **bo, *perfmon;
    unsigned int bo_count;
};
struct drm_v3d_submit_csd {
    u32 cfg[8], pad, flags, perfmon_id, out_sync;
    uint64_t extensions;
};
struct v3d_csd_job { struct v3d_job base; struct drm_v3d_submit_csd args; };
struct v3d_file_priv { int unused; };
struct drm_file { struct v3d_file_priv *driver_priv; };
struct v3d_submit_ext { int unused; };
struct ww_acquire_ctx { int unused; };

static int allocation_error, allocation_calls, replacement_calls;
static int module_refs, log_count, cleanup_count, storage_released;
static int clean_begin, clean_end, wait_count, wait_failure, writes, invalidates;
static int stats_begin, stats_end, fence_error, tests;
static bool poison_during_wait;
static struct v3d_dev *observed_dev;
static struct dma_fence stub_fence = { .status = 1, .refs = 1 };
static struct v3d_fence hardware_fence;
static struct device device;
static struct v3d_dev dev;
static struct drm_sched_fence sched_fence, clean_sched_fence;
static struct v3d_csd_job csd;
static struct v3d_job clean_job;
static int setup_error, setup_calls, push_count, attachment_count, unlock_count;
static bool poison_at_acceptance;
static struct dma_fence *published;

#define READ_ONCE(x) (x)
#define ERR_PTR(error) ((void *)(intptr_t)(error))
#define IS_ERR(ptr) ((uintptr_t)(ptr) >= (uintptr_t)-4095)
#define PTR_ERR(ptr) ((int)(intptr_t)(ptr))
#define GFP_KERNEL 0
#define GFP_NOWAIT 1
#define xa_limit_32b 0
#define THIS_MODULE NULL
#define DRM_V3D_SUBMIT_EXTENSION 1
#define DRM_DEBUG(...) ((void)0)
#define DRM_INFO(...) ((void)0)
#define trace_v3d_submit_csd_ioctl(dev, a, b) ((void)(dev), (void)(a), (void)(b))
#define V3D_CSD 0
#define V3D_CACHE_CLEAN 1
#define V3D_GEN_71 71
#define V3D_CTL_L2TCACTL 100
#define V3D_L2TCACTL_L2TFLS 1
#define V3D_L2TCACTL_TMUWCF 2
#define V3D_L2TCACTL_FLM_CLEAN 4
#define V3D_L2TCACTL_FLM 4
#define V3D_V7_CSD_QUEUED_CFG7 999
#define V3D_CSD_QUEUED_CFG0_NUM_WGS_X 0
#define V3D_CSD_QUEUED_CFG1_NUM_WGS_Y 0
#define V3D_CSD_QUEUED_CFG2_NUM_WGS_Z 0
#define V3D_GET_FIELD(value, field) ((value) >> 16)
#define V3D_SET_FIELD(value, field) (value)
#define V3D_CSD_QUEUED_CFG0(version) 200
#define V3D_CORE_READ(core, reg) ((void)(core), (void)(reg), 0)
#define V3D_CORE_WRITE(core, reg, value) do { (void)(core); (void)(reg); (void)(value); ++writes; } while (0)
#define DRM_ERROR(...) (++log_count)
#define drm_err(dev, ...) ((void)(dev), ++log_count)
#define trace_v3d_cache_clean_begin(dev) ((void)(dev), ++clean_begin)
#define trace_v3d_cache_clean_end(dev) ((void)(dev), ++clean_end)
#define trace_v3d_submit_csd(dev, seqno) ((void)(dev), (void)(seqno))
#define xa_for_each(array, index, entry) \
    for ((index) = 0; (index) < 16; ++(index)) \
        if (((entry) = (array)->entries[index]) != NULL)

static int cmpxchg(int *p, int expected, int replacement) {
    int previous = *p;
    if (previous == expected) *p = replacement;
    return previous;
}
static void kref_get(struct kref *ref) { ++ref->count; }
static void __module_get(void *module) { ++module_refs; }
static void drm_dev_get(struct drm_device *dev) { ++dev->refs; }
static void get_device(struct device *dev) { ++dev->refs; }
static struct dma_fence *dma_fence_get(struct dma_fence *f) { if (f) ++f->refs; return f; }
static void dma_fence_put(struct dma_fence *f) { if (f) { assert(f->refs > 0); --f->refs; } }
static int dma_fence_get_status(struct dma_fence *f) { return f->status; }
static struct dma_fence *dma_fence_get_stub(void) { return dma_fence_get(&stub_fence); }
static int xa_alloc(struct xarray *a, u32 *index, struct dma_fence *f, int limit, int gfp) {
    ++allocation_calls;
    if (allocation_error) return allocation_error;
    for (u32 i = 0; i < 16; ++i) if (!a->entries[i]) { *index = i; a->entries[i] = f; return 0; }
    return -ENOMEM;
}
static struct dma_fence *xa_store(struct xarray *a, u32 index, struct dma_fence *f, int gfp) {
    assert(index < 16 && a->entries[index]);
    assert(gfp == GFP_NOWAIT);
    ++replacement_calls;
    struct dma_fence *old = a->entries[index];
    a->entries[index] = f;
    return old;
}
static struct v3d_job *to_v3d_job(struct drm_sched_job *base) {
    return (struct v3d_job *)((char *)base - offsetof(struct v3d_job, base));
}
static struct v3d_csd_job *to_csd_job(struct drm_sched_job *base) {
    return (struct v3d_csd_job *)((char *)base - offsetof(struct v3d_csd_job, base.base));
}
static struct v3d_fence *to_v3d_fence(struct dma_fence *fence) {
    return (struct v3d_fence *)((char *)fence - offsetof(struct v3d_fence, base));
}
static void v3d_job_cleanup(struct v3d_job *job) {
    if (!job) return;
    ++cleanup_count;
    if (!--job->refcount.count) ++storage_released;
    job->base.s_fence = NULL;
}
static void mutex_lock(int *lock) {
    assert(!*lock); *lock = 1;
    if (poison_at_acceptance && lock == &dev.sched_lock) dev.csd_error = -EIO;
}
static void mutex_unlock(int *lock) { assert(*lock); *lock = 0; }
static int scripted_wait(void) {
    ++wait_count;
    if (poison_during_wait) observed_dev->csd_error = -EFAULT;
    return wait_count == wait_failure ? -ETIMEDOUT : 0;
}
#define wait_for(condition, timeout) ((void)(condition), (void)(timeout), scripted_wait())
static void v3d_job_start_stats(struct v3d_job *job, int queue) { ++stats_begin; }
static void v3d_job_update_stats(struct v3d_job *job, int queue) { ++stats_end; }
static void v3d_invalidate_caches(struct v3d_dev *v3d) { ++invalidates; }
static void v3d_switch_perfmon(struct v3d_dev *v3d, struct v3d_job *job) {}
static struct dma_fence *v3d_fence_create(struct v3d_dev *v3d, int queue) {
    if (fence_error) return ERR_PTR(fence_error);
    hardware_fence.base.refs = 1;
    return &hardware_fence.base;
}

/* Direct ioctl boundaries: count actual acceptance and publication. */
static struct v3d_dev *to_v3d_dev(struct drm_device *drm) {
    return (struct v3d_dev *)((char *)drm - offsetof(struct v3d_dev, drm));
}
static bool v3d_has_csd(struct v3d_dev *v3d) { return v3d->ver >= 41; }
static int v3d_get_extensions(struct drm_file *file, uint64_t ext,
                              struct v3d_submit_ext *se, void *cpu) { return 0; }
static int v3d_setup_csd_jobs_and_bos(struct drm_file *file, struct v3d_dev *v3d,
        struct drm_v3d_submit_csd *args, struct v3d_csd_job **job,
        struct v3d_job **clean, struct v3d_submit_ext *se, struct ww_acquire_ctx *ctx) {
    ++setup_calls;
    if (setup_error) return setup_error;
    *job = &csd; *clean = &clean_job; csd.args = *args;
    return 0;
}
static void *v3d_perfmon_find(struct v3d_file_priv *file, u32 id) { return NULL; }
static void v3d_job_put(struct v3d_job *job) { if (job) --job->refcount.count; }
static void v3d_push_job(struct v3d_job *job) {
    ++push_count;
    job->done_fence = &job->base.s_fence->finished;
    job->done_fence->refs = 1;
    ++job->refcount.count;
}
static void drm_gem_unlock_reservations(void **bos, unsigned int count,
                                         struct ww_acquire_ctx *ctx) { ++unlock_count; }
static void v3d_attach_fences_and_unlock_reservation(struct drm_file *file,
        struct v3d_job *job, struct ww_acquire_ctx *ctx, u32 out_sync,
        struct v3d_submit_ext *se, struct dma_fence *fence) {
    ++attachment_count; ++unlock_count; published = fence;
    assert(job == &clean_job);
    assert(job->base.dependencies.entries[0] == csd.base.done_fence);
}
static void v3d_put_multisync_post_deps(struct v3d_submit_ext *se) {}

#include "extracted.inc"

static void reset_fixture(void) {
    memset(&device, 0, sizeof(device));
    memset(&dev, 0, sizeof(dev)); dev.drm.dev = &device; dev.ver = 71;
    memset(&sched_fence, 0, sizeof(sched_fence));
    memset(&csd, 0, sizeof(csd)); csd.base.v3d = &dev;
    csd.base.base.s_fence = &sched_fence; csd.base.refcount.count = 1;
    csd.args.cfg[0] = csd.args.cfg[1] = csd.args.cfg[2] = 1u << 16;
    allocation_error = allocation_calls = replacement_calls = 0;
    module_refs = log_count = cleanup_count = storage_released = 0;
    clean_begin = clean_end = wait_count = wait_failure = writes = invalidates = 0;
    stats_begin = stats_end = fence_error = 0; poison_during_wait = false;
    observed_dev = &dev;
    memset(&clean_job, 0, sizeof(clean_job));
    memset(&clean_sched_fence, 0, sizeof(clean_sched_fence));
    clean_job.v3d = &dev; clean_job.refcount.count = 1;
    clean_job.base.s_fence = &clean_sched_fence;
    clean_job.needs_cache_clean = true;
    setup_error = setup_calls = push_count = attachment_count = unlock_count = 0;
    poison_at_acceptance = false; published = NULL;
}

static void dependencies(void) {
    reset_fixture(); u32 index = 0; int original = stub_fence.refs;
    allocation_error = -ENOMEM;
    assert(v3d_reserve_chain_dependency(&csd.base, &index) == -ENOMEM);
    assert(stub_fence.refs == original && replacement_calls == 0); ++tests;

    allocation_error = 0;
    assert(v3d_reserve_chain_dependency(&csd.base, &index) == 0);
    assert(stub_fence.refs == original + 1);
    struct dma_fence parent_done = { .status = 0, .refs = 1 };
    struct v3d_job parent = { .done_fence = &parent_done };
    int before = allocation_calls;
    allocation_error = -ENOMEM; /* allocation forbidden after acceptance */
    v3d_set_chain_dependency(&csd.base, index, &parent);
    assert(allocation_calls == before && replacement_calls == 1);
    assert(csd.base.base.dependencies.entries[index] == &parent_done);
    assert(stub_fence.refs == original && parent_done.refs == 2);
    parent_done.status = -EFAULT;
    assert(v3d_job_error(&csd.base) == -EFAULT); ++tests;
    dma_fence_put(&parent_done);

    reset_fixture(); sched_fence.finished.error = -ECANCELED;
    assert(v3d_job_error(&csd.base) == -ECANCELED); ++tests;
    reset_fixture(); dev.reset_counter = 1;
    assert(v3d_job_error(&csd.base) == -EIO); ++tests;
}

static void cache_errors(void) {
    for (int version = 41; version <= 71; version += 30) {
        const int stages = version == 41 ? 3 : 2;
        for (int stage = 1; stage <= stages; ++stage) {
            reset_fixture(); dev.ver = version; wait_failure = stage;
            assert(v3d_clean_caches(&dev) == -ETIMEDOUT);
            assert(wait_count == stage && !dev.cache_clean_lock);
            assert(clean_begin == 1 && clean_end == 1); ++tests;
        }
        reset_fixture(); dev.ver = version;
        assert(v3d_clean_caches(&dev) == 0 && wait_count == stages);
        assert(!dev.cache_clean_lock); ++tests;
    }
}

static void clean_retirement(void) {
    reset_fixture(); csd.base.needs_cache_clean = true;
    assert(v3d_cache_clean_job_run(&csd.base.base) == NULL);
    assert(csd.base.cache_clean_complete && !csd.base.quarantined);
    v3d_sched_job_free(&csd.base.base);
    assert(storage_released == 1 && module_refs == 0 && dev.csd_error == 0); ++tests;

    reset_fixture(); csd.base.needs_cache_clean = true; wait_failure = 2;
    assert(PTR_ERR(v3d_cache_clean_job_run(&csd.base.base)) == -ETIMEDOUT);
    assert(!csd.base.cache_clean_complete && csd.base.quarantined);
    assert(dev.csd_error == -ETIMEDOUT && module_refs == 1);
    assert(dev.drm.refs == 1 && device.refs == 1);
    v3d_sched_job_free(&csd.base.base);
    assert(storage_released == 0 && csd.base.refcount.count == 1);
    assert(module_refs == 1 && dev.csd_error == -ETIMEDOUT); ++tests;

    reset_fixture(); csd.base.needs_cache_clean = true;
    sched_fence.finished.error = -ESRCH; /* scheduler cancelled the clean */
    v3d_sched_job_free(&csd.base.base);
    assert(storage_released == 0 && dev.csd_error == -EIO);
    assert(csd.base.refcount.count == 1 && csd.base.quarantined); ++tests;

    reset_fixture(); csd.base.needs_cache_clean = true; poison_during_wait = true;
    assert(PTR_ERR(v3d_cache_clean_job_run(&csd.base.base)) == -EFAULT);
    assert(!csd.base.cache_clean_complete && csd.base.quarantined); ++tests;

    reset_fixture(); csd.base.needs_cache_clean = true;
    struct dma_fence dep = { .status = -EFAULT, .refs = 1 };
    csd.base.base.dependencies.entries[0] = &dep;
    assert(PTR_ERR(v3d_cache_clean_job_run(&csd.base.base)) == -EFAULT);
    assert(!wait_count && csd.base.quarantined); ++tests;
}

static void csd_error_and_reset(void) {
    reset_fixture(); fence_error = -ENOMEM;
    assert(PTR_ERR(v3d_csd_job_run(&csd.base.base)) == -ENOMEM);
    assert(!writes && !invalidates && !dev.queue[V3D_CSD].active_job); ++tests;

    reset_fixture(); dev.reset_counter = 1;
    assert(PTR_ERR(v3d_csd_job_run(&csd.base.base)) == -EIO);
    assert(!writes && dev.csd_error == -EIO && csd.base.quarantined); ++tests;

    reset_fixture(); struct dma_fence dep = { .status = -EFAULT, .refs = 1 };
    csd.base.base.dependencies.entries[0] = &dep;
    assert(PTR_ERR(v3d_csd_job_run(&csd.base.base)) == -EFAULT);
    assert(!writes && dev.csd_error == -EFAULT); ++tests;

    reset_fixture(); csd.args.cfg[1] = 0;
    assert(v3d_csd_job_run(&csd.base.base) == NULL);
    assert(!writes && !dev.csd_error && !csd.base.quarantined); ++tests;

    reset_fixture(); assert(v3d_csd_job_run(&csd.base.base) == &hardware_fence.base);
    assert(writes == 8 && invalidates == 1 && stats_begin == 1);
    assert(dev.queue[V3D_CSD].active_job == &csd.base && !dev.csd_error);
    dma_fence_put(csd.base.irq_fence); dma_fence_put(&hardware_fence.base); ++tests;

    reset_fixture(); v3d_csd_poison_device(&dev, -ETIMEDOUT);
    v3d_csd_poison_device(&dev, -EFAULT);
    assert(dev.csd_error == -ETIMEDOUT && module_refs == 1);
    assert(dev.drm.refs == 1 && device.refs == 1); ++tests;
}

static void direct_submission(void) {
    struct drm_file file = { 0 };
    struct drm_v3d_submit_csd args = { .cfg = { 1u << 16, 1u << 16, 1u << 16 } };
    reset_fixture(); setup_error = -ENOMEM;
    assert(v3d_submit_csd_ioctl(&dev.drm, &args, &file) == -ENOMEM);
    assert(push_count == 0 && !attachment_count && !published); ++tests;

    reset_fixture(); allocation_error = -ENOMEM;
    assert(v3d_submit_csd_ioctl(&dev.drm, &args, &file) == -ENOMEM);
    assert(push_count == 0 && !attachment_count && cleanup_count == 2);
    assert(storage_released == 2 && unlock_count == 1 && !published); ++tests;

    reset_fixture(); poison_at_acceptance = true;
    assert(v3d_submit_csd_ioctl(&dev.drm, &args, &file) == -EIO);
    assert(push_count == 0 && !attachment_count && !dev.sched_lock); ++tests;

    reset_fixture(); dev.csd_error = -EIO;
    assert(v3d_submit_csd_ioctl(&dev.drm, &args, &file) == -EIO);
    assert(!setup_calls && !push_count && !attachment_count); ++tests;

    reset_fixture();
    assert(v3d_submit_csd_ioctl(&dev.drm, &args, &file) == 0);
    assert(push_count == 2 && attachment_count == 1 && unlock_count == 1);
    assert(published == clean_job.done_fence && !dev.sched_lock);
    assert(csd.base.refcount.count == 1 && clean_job.refcount.count == 1);
    assert(allocation_calls == 1 && replacement_calls == 1); ++tests;
}

int main(void) {
    dependencies(); cache_errors(); clean_retirement(); csd_error_and_reset(); direct_submission();
    printf("{\"cases\": %d, \"status\": \"PASS\"}\n", tests);
    return 0;
}
