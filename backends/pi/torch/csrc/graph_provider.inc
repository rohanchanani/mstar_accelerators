// Actual ggml metadata and prepared graph slots. Included in ggml_provider.cpp
// so linear and graph providers share the same import, tail and error boundary.
struct GraphData {
  ggml_context* context = nullptr;
  ggml_cgraph* graph = nullptr;
  std::size_t capacity;
  std::size_t count = 0;
  bool sealed = false;
  std::map<std::string, ggml_tensor*> inputs;
  std::map<std::string, ggml_tensor*> outputs;
  std::vector<ggml_tensor*> roots;
  explicit GraphData(std::size_t n) : capacity(n) {
    if (n < 8) throw native::Rejected("graph capacity must be at least 8 tensors");
    context = ggml_init({n * ggml_tensor_overhead() + ggml_graph_overhead_custom(n, false), nullptr, true});
    if (!context) throw native::Rejected("ggml graph metadata allocation failed");
    graph = ggml_new_graph_custom(context, n, false);
  }
  ~GraphData() { if (context) ggml_free(context); }
  void room(std::size_t n = 1) {
    if (sealed) throw native::Rejected("prepared ggml graph metadata is immutable");
    if (n > capacity - count) throw native::Rejected("ggml graph metadata capacity exhausted");
    count += n;
  }
  void finish() {
    if (sealed) throw native::Rejected("a graph instance can belong to only one prepared slot");
    if (roots.empty()) throw native::Rejected("graph needs an output or effect root");
    for (auto* t : roots) ggml_build_forward_expand(graph, t);
    sealed = true;
  }
};

struct GraphTensor {
  std::shared_ptr<GraphData> graph;
  ggml_tensor* tensor;
  std::vector<std::int64_t> ne() const { return {tensor->ne, tensor->ne + GGML_MAX_DIMS}; }
  std::vector<std::size_t> nb() const { return {tensor->nb, tensor->nb + GGML_MAX_DIMS}; }
};

struct GraphArenaLayout {
  std::size_t bytes = 0;
  std::vector<std::pair<ggml_tensor*,std::size_t>> tensors;
};

GraphArenaLayout graph_arena_layout(const GraphData& graph, std::size_t alignment) {
  if (!alignment) throw native::Rejected("graph arena alignment must be positive");
  std::set<ggml_tensor*> external;
  for (const auto& [name,tensor] : graph.inputs) external.insert(tensor);
  for (const auto& [name,tensor] : graph.outputs) external.insert(tensor);
  GraphArenaLayout result;
  for (auto* t=ggml_get_first_tensor(graph.context); t; t=ggml_get_next_tensor(graph.context,t)) {
    if (t->view_src || external.contains(t)) continue;
    result.bytes=(result.bytes+alignment-1)/alignment*alignment;
    result.tensors.emplace_back(t,result.bytes);
    result.bytes+=ggml_nbytes(t);
  }
  return result;
}

ggml_type graph_type(const std::string& type) {
  if (type == "f32") return GGML_TYPE_F32;
  if (type == "i32") return GGML_TYPE_I32;
  if (type == "i64") return GGML_TYPE_I64;
  if (type == "q8_0") return GGML_TYPE_Q8_0;
  throw native::Rejected("graph tensor dtype must be f32, i32, i64, or explicit q8_0 constant storage");
}

std::array<std::int64_t, 4> graph_dims(const std::vector<std::int64_t>& ne) {
  if (ne.empty() || ne.size() > 4) throw native::Rejected("ggml tensor needs 1 to 4 dimensions");
  std::array<std::int64_t, 4> dims{1, 1, 1, 1};
  std::uint64_t count = 1;
  for (std::size_t i = 0; i < ne.size(); ++i) {
    if (ne[i] <= 0 || std::uint64_t(ne[i]) > UINT32_MAX / count)
      throw native::Rejected("ggml tensor extent exceeds the graph index domain");
    dims[i] = ne[i]; count *= ne[i];
  }
  return dims;
}

class Graph {
public:
  explicit Graph(std::size_t max_nodes = 4096) : data(std::make_shared<GraphData>(max_nodes)) {}
  std::shared_ptr<GraphData> data;
  ggml_tensor* take(const GraphTensor& t, bool allow_q8 = false) const {
    if (t.graph != data) throw native::Rejected("ggml tensor belongs to a different graph");
    if (!allow_q8 && t.tensor->type == GGML_TYPE_Q8_0)
      throw native::Rejected("Q8_0 constants are readonly operands of q8_0_dot_x4 only");
    return t.tensor;
  }
  GraphTensor wrap(ggml_tensor* t) const { return {data, t}; }
  GraphTensor tensor(const std::string& name, const std::vector<std::int64_t>& ne, const std::string& type) {
    const auto dims = graph_dims(ne);
    const auto dtype = graph_type(type);
    if (dtype == GGML_TYPE_Q8_0 && (dims[0] % 32 || dims[2] != 1 || dims[3] != 1 ||
        (uint64_t(dims[0]) / 32 * uint64_t(dims[1]) * 34) % sizeof(int32_t)))
      throw native::Rejected("Q8_0 constants require a block-aligned matrix with exact I32 byte extent");
    if (data->inputs.contains(name) || data->outputs.contains(name)) throw native::Rejected("duplicate graph binding name");
    data->room();
    auto* t = ggml_new_tensor(data->context, dtype, 4, dims.data());
    ggml_set_name(t, name.c_str());
    data->inputs.emplace(name, t);
    return wrap(t);
  }
  GraphTensor binary(const GraphTensor& a, const GraphTensor& b, bool multiply) {
    auto* x = take(a); auto* y = take(b);
    if (x->type != GGML_TYPE_F32 || y->type != GGML_TYPE_F32 || !ggml_can_repeat(y, x))
      throw native::Rejected("F32 binary graph operands cannot broadcast");
    data->room();
    return wrap(multiply ? ggml_mul(data->context, x, y) : ggml_add(data->context, x, y));
  }
  GraphTensor mul_mat(const GraphTensor& a, const GraphTensor& b) {
    auto* x = take(a); auto* y = take(b);
    if (x->type != GGML_TYPE_F32 || y->type != GGML_TYPE_F32 || x->ne[0] != y->ne[0] ||
        y->ne[2] % x->ne[2] || y->ne[3] % x->ne[3])
      throw native::Rejected("invalid F32 matrix dimensions/broadcast");
    data->room();
    auto* t = ggml_mul_mat(data->context, x, y);
    if (!ggml_prec_set_acc(t, GGML_PREC_F32) || !ggml_prec_set_src(t, GGML_PREC_F32, 1))
      throw native::Rejected("ggml cannot represent F32 matrix precision");
    return wrap(t);
  }
  GraphTensor quantize_q8_1_x4(const GraphTensor& a) {
    auto* x = take(a); data->room();
    auto* result = ggml_vk_managed_quantize_q8_1_x4(data->context, x);
    if (!result) throw native::Rejected(ggml_vk_managed_last_error());
    return wrap(result);
  }
  GraphTensor q8_0_dot_x4(const GraphTensor& weights, const GraphTensor& packed) {
    auto* w = take(weights, true); auto* q = take(packed); data->room();
    auto* result = ggml_vk_managed_q8_0_dot_x4(data->context, w, q);
    if (!result) throw native::Rejected(ggml_vk_managed_last_error());
    return wrap(result);
  }
  GraphTensor sdpa_decode_f32(const GraphTensor& q, const GraphTensor& k,
                            const GraphTensor& v, const GraphTensor& mask, float scale) {
    auto* query = take(q); auto* key = take(k); auto* value = take(v); auto* masking = take(mask);
    data->room();
    auto* result = ggml_vk_managed_sdpa_decode_f32(data->context, query, key, value, masking, scale);
    if (!result) throw native::Rejected(std::string("SDPA metadata: ") + ggml_vk_managed_last_error());
    return wrap(result);
  }
  GraphTensor sdpa_queries_f32(const GraphTensor& q, const GraphTensor& k,
                            const GraphTensor& v, const GraphTensor& mask, float scale) {
    auto* query = take(q); auto* key = take(k); auto* value = take(v); auto* masking = take(mask);
    data->room();
    auto* result = ggml_vk_managed_sdpa_queries_f32(data->context, query, key, value, masking, scale);
    if (!result) throw native::Rejected(std::string("Multi-query SDPA metadata: ") + ggml_vk_managed_last_error());
    return wrap(result);
  }
  GraphTensor scale(const GraphTensor& a, float value) {
    auto* t = take(a); data->room(); return wrap(ggml_scale(data->context, t, value));
  }
  GraphTensor norm(const GraphTensor& a, float eps) {
    auto* t = take(a); data->room(); return wrap(ggml_norm(data->context, t, eps));
  }
  GraphTensor gelu_erf(const GraphTensor& a) {
    auto* t = take(a); data->room(); return wrap(ggml_gelu_erf(data->context, t));
  }
  GraphTensor softmax(const GraphTensor& a, const std::optional<GraphTensor>& mask, float scale) {
    auto* t = take(a); auto* m = mask ? take(*mask) : nullptr;
    if (m && (m->type != GGML_TYPE_F32 || m->ne[0] != t->ne[0] || m->ne[1] < t->ne[1] ||
              t->ne[2] % m->ne[2] || t->ne[3] % m->ne[3]))
      throw native::Rejected("softmax mask does not cover graph rows");
    data->room(); return wrap(ggml_soft_max_ext(data->context, t, m, scale, 0.f));
  }
  GraphTensor reshape(const GraphTensor& a, const std::vector<std::int64_t>& ne) {
    auto* t = take(a); const auto d = graph_dims(ne);
    if (!ggml_is_contiguous(t) || ggml_nelements(t) != d[0]*d[1]*d[2]*d[3])
      throw native::Rejected("reshape requires equal contiguous element count");
    data->room(); return wrap(ggml_reshape_4d(data->context, t, d[0], d[1], d[2], d[3]));
  }
  GraphTensor permute(const GraphTensor& a, const std::vector<int>& axes) {
    auto* t = take(a); auto sorted = axes; std::sort(sorted.begin(), sorted.end());
    if (sorted != std::vector<int>{0, 1, 2, 3}) throw native::Rejected("ggml permutation requires axes 0,1,2,3");
    data->room(); return wrap(ggml_permute(data->context, t, axes[0], axes[1], axes[2], axes[3]));
  }
  GraphTensor cont(const GraphTensor& a) {
    auto* t = take(a); data->room(); return wrap(ggml_cont(data->context, t));
  }
  GraphTensor view(const GraphTensor& a, const std::vector<std::int64_t>& ne,
                   const std::vector<std::size_t>& strides, std::size_t offset) {
    auto* t = take(a); const auto d = graph_dims(ne);
    if (strides.size() != ne.size() || strides[0] != ggml_type_size(t->type))
      throw native::Rejected("view needs byte strides with contiguous first dimension");
    std::array<std::size_t, 4> nb{strides[0], 0, 0, 0};
    for (std::size_t i=1; i<4; ++i) nb[i] = i < strides.size() ? strides[i] : nb[i-1]*d[i-1];
    std::size_t bytes = ggml_type_size(t->type);
    for (int i=0; i<4; ++i) bytes += (d[i]-1)*nb[i];
    if (offset > ggml_nbytes(t) || bytes > ggml_nbytes(t)-offset)
      throw native::Rejected("view exceeds its graph tensor");
    data->room(); return wrap(ggml_view_4d(data->context, t, d[0], d[1], d[2], d[3], nb[1], nb[2], nb[3], offset));
  }
  GraphTensor get_rows(const GraphTensor& a, const GraphTensor& indices) {
    auto* t = take(a); auto* i = take(indices);
    if (i->type != GGML_TYPE_I32 || t->ne[2] != i->ne[1] || t->ne[3] != i->ne[2] || i->ne[3] != 1)
      throw native::Rejected("get_rows requires I32 indices matching batch dimensions");
    data->room(); return wrap(ggml_get_rows(data->context, t, i));
  }
  GraphTensor set_rows(const GraphTensor& a, const GraphTensor& indices, const GraphTensor& destination) {
    auto* src = take(a); auto* idx = take(indices); auto* dst = take(destination);
    if ((idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64) || src->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32 || dst->ne[0] != src->ne[0] || dst->ne[2] != src->ne[2] ||
        dst->ne[3] != src->ne[3] || src->ne[1] != idx->ne[0] || src->ne[2] % idx->ne[1] ||
        src->ne[3] % idx->ne[2] || idx->ne[3] != 1 || !ggml_is_contiguous_rows(dst) || !ggml_is_contiguous_rows(src))
      throw native::Rejected("set_rows requires matching F32 rows and I32/I64 row indices");
    data->room(); return wrap(ggml_set_rows(data->context, dst, src, idx));
  }
  GraphTensor im2col(const GraphTensor& kernel, const GraphTensor& x, int stride, int padding, int dilation) {
    auto* k = take(kernel); auto* t = take(x);
    if (k->type != GGML_TYPE_F32 || t->type != GGML_TYPE_F32 || stride <= 0 || padding < 0 || dilation <= 0 ||
        k->ne[1] != t->ne[1] || k->ne[3] != 1 || t->ne[3] != 1 ||
        t->ne[0]+2*padding-dilation*(k->ne[0]-1)-1 < 0)
      throw native::Rejected("invalid 1D F32 im2col geometry");
    data->room(); return wrap(ggml_im2col(data->context, k, t, stride, 1, padding, 0, dilation, 1, false, GGML_TYPE_F32));
  }
  GraphTensor argmax(const GraphTensor& a) {
    auto* t = take(a); data->room(); return wrap(ggml_argmax(data->context, t));
  }
  GraphTensor categorical(const GraphTensor& logits, const GraphTensor& seeds,
                          const GraphTensor& offsets, const GraphTensor& temperatures) {
    auto* l = take(logits); auto* s = take(seeds); auto* o = take(offsets); auto* t = take(temperatures);
    if (l->type != GGML_TYPE_F32 || !ggml_is_matrix(l) || !ggml_is_contiguous(l) ||
        l->ne[0] <= 0 || l->ne[0] > INT32_MAX || l->ne[1] <= 0 || ggml_nelements(l) > UINT32_MAX ||
        s->type != GGML_TYPE_I64 || o->type != GGML_TYPE_I64 || t->type != GGML_TYPE_F32 ||
        !ggml_is_vector(s) || !ggml_is_vector(o) || !ggml_is_vector(t) ||
        !ggml_is_contiguous(s) || !ggml_is_contiguous(o) || !ggml_is_contiguous(t) ||
        s->ne[0] != l->ne[1] || o->ne[0] != l->ne[1] || t->ne[0] != l->ne[1])
      throw native::Rejected("categorical requires contiguous F32 [vocabulary, rows], I64 seed/offset and F32 temperature vectors");
    data->room(); return wrap(ggml_categorical(data->context, l, s, o, t));
  }
  GraphTensor cast(const GraphTensor& a, const std::string& type) {
    auto* t = take(a); const auto dt = graph_type(type);
    if (dt == GGML_TYPE_Q8_0) throw native::Rejected("Q8_0 is an explicitly packed prepared constant, not a cast");
    data->room(); return wrap(ggml_cast(data->context, t, dt));
  }
  GraphTensor output(const std::string& name, const GraphTensor& value) {
    auto* t = take(value);
    if (data->outputs.contains(name) || data->inputs.contains(name)) throw native::Rejected("duplicate graph output name");
    data->room(2);
    auto* dst = ggml_dup_tensor(data->context, t);
    ggml_set_name(dst, name.c_str());
    data->outputs.emplace(name, dst);
    auto* root = ggml_cpy(data->context, t, dst);
    data->roots.push_back(root);
    return wrap(root);
  }
  void effect(const GraphTensor& t) { data->room(0); data->roots.push_back(take(t)); }
  py::dict allocation_plan(std::size_t alignment) const {
    const auto layout=graph_arena_layout(*data,alignment);
    py::dict result;
    result["alignment"]=alignment; result["arena_bytes_per_slot"]=layout.bytes;
    result["intermediate_tensors"]=layout.tensors.size();
    result["backend_scratch_bytes_per_slot"]=0;
    py::dict by_operation;
    std::map<std::string,std::size_t> totals;
    for (const auto& [tensor,offset] : layout.tensors) totals[ggml_op_name(tensor->op)]+=ggml_nbytes(tensor);
    for (const auto& [name,bytes] : totals) by_operation[py::str(name)]=bytes;
    result["intermediate_bytes_by_operation"]=by_operation;
    return result;
  }
  py::dict info() const {
    py::dict result; result["tensor_count"] = data->count; result["sealed"] = data->sealed;
    py::dict inputs, outputs;
    for (const auto& [name,t] : data->inputs) inputs[py::str(name)] = GraphTensor{data,t}.ne();
    for (const auto& [name,t] : data->outputs) outputs[py::str(name)] = GraphTensor{data,t}.ne();
    result["inputs"] = inputs; result["outputs"] = outputs; return result;
  }
};

struct GraphSlot {
  ggml_vk_managed_t managed = nullptr;
  std::shared_ptr<GraphData> graph;
  bool busy = false;
  std::map<std::uint64_t, std::weak_ptr<Import>> imports;
  std::vector<std::shared_ptr<Import>> persistent;
  std::vector<native::View> uses;
  std::vector<ggml_tensor*> mutable_tensors;
  std::map<std::string, ggml_tensor*> inputs;
  native::View arena;
};
struct GraphSlotDeleter {
  void operator()(GraphSlot* slot) const noexcept {
    if (slot->managed && ggml_vk_managed_destroy(slot->managed) != 0) return;
    delete slot;
  }
};
using GraphSlotPtr = std::unique_ptr<GraphSlot, GraphSlotDeleter>;
struct GraphProgram {
  std::shared_ptr<native::Runtime> runtime;
  mutable std::mutex mutex;
  std::vector<GraphSlotPtr> slots;
  std::size_t arena_bytes = 0;
  std::uint64_t submissions = 0, retirements = 0;
  std::shared_ptr<Import> import(GraphSlot& slot, const native::View& view) {
    std::erase_if(slot.imports, [](const auto& e){ return e.second.expired(); });
    const auto id = view.allocation->identity();
    if (auto found = slot.imports.find(id); found != slot.imports.end()) {
      if (auto value = found->second.lock()) return value;
    }
    auto result = std::make_shared<Import>();
    result->whole = {view.allocation, 0, view.allocation->size()};
    OwnedFd fd(runtime->export_dma_buf(result->whole));
    result->buffer = ggml_vk_managed_import_dma_buf(slot.managed, fd.value, result->whole.bytes, view.allocation->backing_size());
    if (!result->buffer) provider_error("graph DMA-BUF import", runtime.get());
    slot.imports[id] = result;
    return result;
  }
};

void check_tensor_shape(const at::Tensor& tensor, const ggml_tensor* spec) {
  if (spec->type == GGML_TYPE_Q8_0) {
    if (tensor.scalar_type() != at::kInt || !tensor.is_contiguous() || tensor.dim() != 1 ||
        uint64_t(tensor.numel()) * sizeof(int32_t) != ggml_nbytes(spec))
      throw native::Rejected("Q8_0 constant requires an exact flat contiguous I32 packed byte container");
    return;
  }
  const auto dtype = spec->type == GGML_TYPE_F32 ? at::kFloat : spec->type == GGML_TYPE_I32 ? at::kInt : at::kLong;
  if (tensor.scalar_type() != dtype || !tensor.is_contiguous() || tensor.dim() > 4)
    throw native::Rejected("graph binding requires matching contiguous dtype/rank");
  std::array<std::int64_t,4> actual{1,1,1,1};
  for (std::int64_t i=0; i<tensor.dim(); ++i) actual[i] = tensor.size(tensor.dim()-1-i);
  if (!std::equal(actual.begin(), actual.end(), spec->ne))
    throw native::Rejected(std::string("graph binding shape mismatch: ") + spec->name);
}

void initialize_views(GraphSlot& slot) {
  for (auto* t = ggml_get_first_tensor(slot.graph->context); t; t = ggml_get_next_tensor(slot.graph->context,t)) {
    if (!t->view_src) continue;
    clear_binding(t);
    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) throw native::Rejected("ggml graph view binding failed");
  }
}

struct GraphFlight final : native::ProviderSubmission {
  std::shared_ptr<GraphProgram> program;
  GraphSlot* slot;
  std::vector<std::shared_ptr<Import>> imports;
  std::vector<native::View> uses;
  std::shared_ptr<Tail> tail = std::make_shared<Tail>();
  OwnedFd unreturned_fd;
  bool accepted = false, retired = false;
  GraphFlight(std::shared_ptr<GraphProgram> p, GraphSlot* s) : program(std::move(p)), slot(s), uses(s->uses) {}
  ~GraphFlight() override {
    if (accepted && !retired) std::terminate();
    // All derived views into a borrowed input/output are rebound on the next
    // lease. Clear them before dropping the invocation's allocation anchors.
    for (auto* t : slot->mutable_tensors) clear_binding(t);
    imports.clear(); uses.clear();
    std::lock_guard lock(program->mutex); slot->busy = false;
  }
  struct Acceptance { native::SubmissionAcceptance* native; GraphFlight* flight; };
  static void may_submit(void* opaque) noexcept {
    auto& marker = *static_cast<Acceptance*>(opaque);
    marker.flight->accepted = true; marker.native->may_submit();
  }
  native::ProviderReceipt submit(int dependency_fd, native::SubmissionAcceptance& acceptance) override {
    Acceptance marker{&acceptance,this};
    check_status(ggml_vk_managed_submit(slot->managed, slot->graph->graph, dependency_fd,
        may_submit, &marker, &tail->value, &unreturned_fd.value), "managed graph submit");
    if (!accepted || !tail->value || unreturned_fd.value < -1)
      throw native::FatalError("managed graph returned an incomplete receipt");
    return {unreturned_fd.release(), tail};
  }
  void retire() override {
    check_status(ggml_vk_managed_retire(slot->managed), "managed graph retirement");
    retired = true;
    std::lock_guard lock(program->mutex); ++program->retirements;
  }
};

class GraphLease {
  std::shared_ptr<GraphProgram> program_;
  GraphSlot* slot_;
public:
  GraphLease(std::shared_ptr<GraphProgram> p, GraphSlot* slot) : program_(std::move(p)), slot_(slot) {}
  ~GraphLease() { close(); }
  void close() noexcept {
    if (slot_) {
      std::lock_guard lock(program_->mutex); slot_->busy = false; slot_ = nullptr;
    }
    program_.reset();
  }
  void submit(const std::map<std::string,at::Tensor>& bindings, const std::map<std::string,at::Tensor>& outputs) {
    if (!slot_) throw native::Rejected("graph lease is closed or already submitted");
    const auto execution = current_execution();
    if (execution.runtime != program_->runtime) throw native::Rejected("graph lease belongs to another runtime");
    // Construct the complete ownership object before releasing the lease. A
    // rejection during import/recording releases it; acceptance transfers it to
    // Runtime Pending, including every failure whose physical status is unknown.
    auto flight = std::make_shared<GraphFlight>(program_, slot_);
    auto* slot = std::exchange(slot_, nullptr);
    if (bindings.size() != slot->inputs.size() || outputs.size() != slot->graph->outputs.size())
      throw native::Rejected("graph invocation binding names do not match its prepared signature");
    for (const auto& [name,spec] : slot->inputs) {
      auto found = bindings.find(name);
      if (found == bindings.end()) throw native::Rejected("missing graph input: " + name);
      check_tensor_shape(found->second, spec);
      const auto view = tensor_view(found->second);
      auto imported = program_->import(*slot,view);
      bind_tensor(spec,imported,view);
      flight->imports.push_back(imported); flight->uses.push_back(view);
    }
    std::vector<native::View> output_views;
    for (const auto& [name,spec] : slot->graph->outputs) {
      auto found = outputs.find(name);
      if (found == outputs.end()) throw native::Rejected("missing graph output: " + name);
      check_tensor_shape(found->second,spec);
      const auto view = tensor_view(found->second);
      for (const auto& other : flight->uses) if (overlaps(view,other))
        throw native::Rejected("graph output overlaps live input, persistent state or workspace");
      for (const auto& other : output_views) if (overlaps(view,other))
        throw native::Rejected("graph outputs overlap");
      auto imported = program_->import(*slot,view);
      bind_tensor(spec,imported,view);
      flight->imports.push_back(imported); flight->uses.push_back(view); output_views.push_back(view);
    }
    initialize_views(*slot);
    program_->runtime->submit_provider(execution.stream, flight->uses, flight);
    {
      std::lock_guard lock(program_->mutex); ++program_->submissions;
    }
    program_.reset();
  }
};

class PreparedGraph {
  std::shared_ptr<GraphProgram> program_;
public:
  PreparedGraph(const py::function& factory, const std::map<std::string,at::Tensor>& constants, std::size_t count) {
    const auto execution = current_execution();
    if (count == 0) throw native::Rejected("prepared graph needs at least one mutable slot");
    auto program = std::make_shared<GraphProgram>(); program->runtime = execution.runtime;
    const auto device = program->runtime->device_identity();
    for (std::size_t index=0; index<count; ++index) {
      auto graph = factory().cast<Graph>(); graph.data->finish();
      GraphSlotPtr slot(new GraphSlot); slot->graph = graph.data;
      slot->managed = ggml_vk_managed_create(device.major,device.minor);
      if (!slot->managed) provider_error("managed graph context",program->runtime.get());
      check_status(ggml_vk_managed_prepare(slot->managed,graph.data->graph),"managed graph prepare",program->runtime.get());
      for (const auto& [name,spec] : graph.data->inputs) {
        if (auto found=constants.find(name); found != constants.end()) {
          check_tensor_shape(found->second,spec);
          const auto view=tensor_view(found->second); auto imported=program->import(*slot,view);
          bind_tensor(spec,imported,view); slot->persistent.push_back(imported); slot->uses.push_back(view);
        } else {
          if (spec->type == GGML_TYPE_Q8_0)
            throw native::Rejected("Q8_0 graph input must be supplied as a prepared readonly constant");
          slot->inputs.emplace(name,spec); slot->mutable_tensors.push_back(spec);
        }
      }
      for (const auto& [name,spec] : graph.data->outputs) {
        slot->mutable_tensors.push_back(spec);
      }
      // A single canonical native arena per exclusive slot. The monotonic
      // allocator is intentional: no hidden alias optimization or scratch
      // lifetime proof is required. A liveness allocator may replace only this
      // allocation policy later, without changing graph/lease/completion APIs.
      const auto backend=ggml_vk_managed_backend(slot->managed);
      const auto alignment=ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
      const auto arena_layout=graph_arena_layout(*graph.data,alignment);
      for (auto* t=ggml_get_first_tensor(graph.data->context); t; t=ggml_get_next_tensor(graph.data->context,t)) {
        if (t->view_src) slot->mutable_tensors.push_back(t);
      }
      if (arena_layout.bytes) {
        auto allocation=program->runtime->allocate(arena_layout.bytes);
        slot->arena={allocation,0,arena_layout.bytes}; auto imported=program->import(*slot,slot->arena);
        slot->persistent.push_back(imported); slot->uses.push_back(slot->arena);
        for (const auto& [tensor,offset] : arena_layout.tensors) bind_tensor(tensor,imported,{allocation,offset,ggml_nbytes(tensor)});
      }
      program->arena_bytes+=arena_layout.bytes;
      program->slots.push_back(std::move(slot));
    }
    program_=std::move(program);
  }
  void close() {
    if (!program_) return;
    // Explicit teardown is outside scheduling. Drain before destroying imported
    // arenas/descriptor state; failures leave the ownership object intact.
    program_->runtime->synchronize_device();
    program_->runtime->collect();
    {
      std::lock_guard lock(program_->mutex);
      for (const auto& slot : program_->slots) if (slot->busy)
        throw native::Rejected("close requires returning unsubmitted graph leases");
    }
    program_.reset();
  }
  bool available() {
    if (!program_) return false;
    program_->runtime->collect();
    std::lock_guard lock(program_->mutex);
    return std::any_of(program_->slots.begin(),program_->slots.end(),[](const auto& s){return !s->busy;});
  }
  std::shared_ptr<GraphLease> acquire() {
    if (!program_) throw native::Rejected("prepared graph is closed");
    program_->runtime->collect();
    std::lock_guard lock(program_->mutex);
    for (const auto& slot : program_->slots) if (!slot->busy) {
      auto lease=std::make_shared<GraphLease>(program_,slot.get()); slot->busy=true; return lease;
    }
    return {};
  }
  py::dict info() const {
    if (!program_) throw native::Rejected("prepared graph is closed");
    std::lock_guard lock(program_->mutex); py::dict info;
    std::size_t busy=0; for(const auto& s:program_->slots) busy+=s->busy;
    info["provider"]="ggml-vulkan-managed"; info["slots"]=program_->slots.size(); info["active_slots"]=busy;
    info["arena_bytes_total"]=program_->arena_bytes; info["backend_scratch_bytes_per_slot"]=0;
    info["submissions"]=program_->submissions; info["retirements"]=program_->retirements;
    info["implicit_host_copies"]=false;
    bool q8 = false;
    for (const auto& [name,tensor] : program_->slots.front()->graph->inputs)
      q8 = q8 || tensor->type == GGML_TYPE_Q8_0;
    info["source_precision"] = q8 ? "explicit-q8_0/q8_1_x4-with-f32-output" : "float32";
    auto* graph=program_->slots.front()->graph->graph;
    info["graph_nodes"]=ggml_graph_n_nodes(graph);
    std::size_t attention=0, queries_attention=0, q8_dots=0, quantizers=0;
    for (int i=0; i<ggml_graph_n_nodes(graph); ++i) {
      const auto* node=ggml_graph_node(graph,i);
      attention += ggml_vk_managed_sdpa_decode_metadata(node,nullptr);
      queries_attention += ggml_vk_managed_sdpa_queries_metadata(node,nullptr);
      const auto kind=ggml_vk_managed_q8_operation(node);
      q8_dots += kind == GGML_VK_MANAGED_Q8_0_DOT_X4;
      quantizers += kind == GGML_VK_MANAGED_QUANTIZE_Q8_1_X4;
    }
    info["managed_sdpa_decode_nodes_per_slot"]=attention;
    info["managed_sdpa_queries_nodes_per_slot"]=queries_attention;
    info["managed_q8_dot_nodes_per_slot"]=q8_dots;
    info["managed_q8_quantize_nodes_per_slot"]=quantizers;
    return info;
  }
};

void register_graph(py::module_& module) {
  py::class_<GraphTensor>(module,"GraphTensor")
    .def_property_readonly("ne",&GraphTensor::ne).def_property_readonly("nb",&GraphTensor::nb);
  py::class_<Graph>(module,"Graph")
    .def(py::init<std::size_t>(),py::arg("max_nodes")=4096)
    .def("tensor",&Graph::tensor,py::arg("name"),py::arg("ne"),py::arg("dtype")="f32")
    .def("mul_mat",&Graph::mul_mat)
    .def("quantize_q8_1_x4",&Graph::quantize_q8_1_x4)
    .def("q8_0_dot_x4",&Graph::q8_0_dot_x4)
    .def("add",[](Graph& g,const GraphTensor& a,const GraphTensor& b){return g.binary(a,b,false);})
    .def("mul",[](Graph& g,const GraphTensor& a,const GraphTensor& b){return g.binary(a,b,true);})
    .def("scale",&Graph::scale).def("norm",&Graph::norm,py::arg("a"),py::arg("eps")=1e-5f)
    .def("gelu_erf",&Graph::gelu_erf)
    .def("softmax",&Graph::softmax,py::arg("a"),py::arg("mask")=std::nullopt,py::arg("scale")=1.f)
    .def("sdpa_decode_f32", &Graph::sdpa_decode_f32)
    .def_static("supports_sdpa_decode_f32",[](std::int64_t heads,std::int64_t tokens,std::int64_t batch) {
      // Metadata rejection precedes device selection. Resolve only the existing
      // initialized Runtime's actual DRM device; query stays in preparation.
      Graph graph;
      const auto q=graph.tensor("q",{64,heads,1,batch},"f32");
      const auto k=graph.tensor("k",{64,heads,tokens,batch},"f32");
      const auto v=graph.tensor("v",{64,heads,tokens,batch},"f32");
      const auto mask=graph.tensor("mask",{tokens,1,1,batch},"f32");
      const auto result=graph.sdpa_decode_f32(q,k,v,mask,.125f);
      const auto execution=current_execution();
      const auto device=execution.runtime->device_identity();
      GraphSlotPtr slot(new GraphSlot);
      slot->managed=ggml_vk_managed_create(device.major,device.minor);
      if (!slot->managed) provider_error("SDPA capability context",execution.runtime.get());
      int supported=0;
      check_status(ggml_vk_managed_supports_sdpa_decode_f32(slot->managed,result.tensor,&supported),
                   "SDPA capability query",execution.runtime.get());
      check_status(ggml_vk_managed_destroy(slot->managed),"SDPA capability close",execution.runtime.get());
      slot->managed=nullptr;
      return supported != 0;
    },py::arg("heads"),py::arg("tokens"),py::arg("batch")=1)
    .def("sdpa_queries_f32", &Graph::sdpa_queries_f32)
    .def_static("supports_sdpa_queries_f32",[](std::int64_t heads,std::int64_t tokens,std::int64_t queries,std::int64_t batch) {
      // Metadata rejection precedes device selection. Resolve only the existing
      // initialized Runtime's actual DRM device; query stays in preparation.
      Graph graph;
      const auto q=graph.tensor("q",{64,heads,queries,batch},"f32");
      const auto k=graph.tensor("k",{64,heads,tokens,batch},"f32");
      const auto v=graph.tensor("v",{64,heads,tokens,batch},"f32");
      const auto mask=graph.tensor("mask",{tokens,queries,1,batch},"f32");
      const auto result=graph.sdpa_queries_f32(q,k,v,mask,.125f);
      const auto execution=current_execution();
      const auto device=execution.runtime->device_identity();
      GraphSlotPtr slot(new GraphSlot);
      slot->managed=ggml_vk_managed_create(device.major,device.minor);
      if (!slot->managed) provider_error("SDPA capability context",execution.runtime.get());
      int supported=0;
      check_status(ggml_vk_managed_supports_sdpa_queries_f32(slot->managed,result.tensor,&supported),
                   "SDPA capability query",execution.runtime.get());
      check_status(ggml_vk_managed_destroy(slot->managed),"SDPA capability close",execution.runtime.get());
      slot->managed=nullptr;
      return supported != 0;
    },py::arg("heads"),py::arg("tokens"),py::arg("queries"),py::arg("batch")=1)
    .def("reshape",&Graph::reshape).def("permute",&Graph::permute).def("cont",&Graph::cont)
    .def("view",&Graph::view,py::arg("a"),py::arg("ne"),py::arg("strides_bytes"),py::arg("offset")=0)
    .def("get_rows",&Graph::get_rows).def("set_rows",&Graph::set_rows)
    .def("im2col",&Graph::im2col,py::arg("kernel"),py::arg("x"),py::arg("stride")=1,py::arg("padding")=0,py::arg("dilation")=1)
    .def("argmax",&Graph::argmax).def("cast",&Graph::cast)
    .def("categorical",&Graph::categorical,py::arg("logits"),py::arg("seeds"),py::arg("offsets"),py::arg("temperatures"))
    .def("output",&Graph::output).def("effect",&Graph::effect).def("info",&Graph::info)
    .def("allocation_plan",&Graph::allocation_plan,py::arg("alignment"));
  py::class_<GraphLease,std::shared_ptr<GraphLease>>(module,"GraphLease")
    .def("submit",&GraphLease::submit,py::arg("bindings"),py::arg("outputs"))
    .def("close",&GraphLease::close).def("release",&GraphLease::close);
  py::class_<PreparedGraph,std::shared_ptr<PreparedGraph>>(module,"PreparedGraph")
    .def(py::init<const py::function&,const std::map<std::string,at::Tensor>&,std::size_t>(),
         py::arg("factory"),py::arg("constants"),py::arg("slots")=2)
    .def("available",&PreparedGraph::available)
    .def("try_acquire",&PreparedGraph::acquire).def("acquire",&PreparedGraph::acquire)
    .def("info",&PreparedGraph::info).def("close",&PreparedGraph::close);
}
