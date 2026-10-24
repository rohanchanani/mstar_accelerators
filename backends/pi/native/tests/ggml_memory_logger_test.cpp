#include "ggml-vulkan-common.h"

#include <sstream>

namespace {
int buffer_destroys;
int memory_frees;

class RecordingBuffer : public std::stringbuf {
public:
    std::vector<std::string> writes;

    std::streamsize xsputn(const char * text, std::streamsize size) override {
        writes.emplace_back(text, size);
        return std::stringbuf::xsputn(text, size);
    }
};

struct InterleavedMessage {};

std::ostream & operator<<(std::ostream & output, InterleavedMessage) {
    std::cerr << "another producer\n";
    return output << "payload";
}

VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice, VkBuffer, const VkAllocationCallbacks *) {
    ++buffer_destroys;
}

VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks *) {
    ++memory_frees;
}
}

int main(int argc, char ** argv) {
    // Exercise real wrapper release, final destruction and logging without a GPU.
    auto & dispatch = ggml_vk_default_dispatcher();
    dispatch.vkDestroyBuffer = destroy_buffer;
    dispatch.vkFreeMemory = free_memory;

    auto device = std::make_shared<vk_device_struct>();
    device->name = "shared-import-test";
    device->memory_logger = std::make_unique<vk_memory_logger>();
    auto buffer = std::make_shared<vk_buffer_struct>();
    buffer->device = device;
    buffer->buffer = vk::Buffer(reinterpret_cast<VkBuffer>(uintptr_t{1}));
    buffer->device_memory = vk::DeviceMemory(reinterpret_cast<VkDeviceMemory>(uintptr_t{2}));
    buffer->memory_property_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    buffer->size = 4096;
    buffer->managed_external = true;

    std::vector<std::string> failures;
    auto check = [&](bool condition, const char * message) {
        if (!condition) failures.emplace_back(message);
    };
    std::ostringstream output;
    auto * previous = std::cerr.rdbuf(output.rdbuf());
    vk_memory_logger_enabled = true;
    device->memory_logger->log_allocation(buffer, buffer->size);
    auto other_slot = buffer;
    auto flight = buffer;

    ggml_vk_destroy_buffer(buffer);
    check(output.str().find(": -") == std::string::npos,
          "releasing one wrapper decremented a shared allocation");
    ggml_vk_destroy_buffer(other_slot);
    check(buffer_destroys == 0 && memory_frees == 0,
          "an outstanding flight did not retain the allocation");
    check(output.str().find(": -") == std::string::npos,
          "releasing the second wrapper decremented a retained allocation");

    flight.reset();
    check(buffer_destroys == 1 && memory_frees == 1 && device->buffer_destroy_count == 1,
          "final reference did not destroy exactly one buffer and memory object");
    check(output.str().find(": -4.00 KiB device") != std::string::npos &&
          output.str().find("Total device: 0 B, total host: 0 B") != std::string::npos,
          "final reference did not retire the allocation from the logger");
    check(output.str().find("ERROR") == std::string::npos,
          "shared wrapper release produced an unknown-allocation error");

    auto partial = std::make_shared<vk_buffer_struct>();
    partial->device = device;
    partial->size = 64;
    partial->managed_external = true;
    const auto logged = output.str();
    partial.reset();
    check(output.str() == logged, "a failed, unlogged import changed the ledger");

    RecordingBuffer framed;
    std::cerr.rdbuf(&framed);
    VK_LOG_MEMORY("record-" << 7 << ": " << 3.5);
    check(framed.writes.size() == 1 &&
          framed.writes.front() == "ggml_vulkan memory: record-7: 3.5\n",
          "a memory record was split into multiple output writes");

    framed.str("");
    framed.writes.clear();
    VK_LOG_MEMORY(InterleavedMessage{});
    check(framed.str() == "another producer\nggml_vulkan memory: payload\n",
          "a memory prefix escaped before the record finished formatting");

    vk_memory_logger_enabled = false;
    std::cerr.rdbuf(previous);
    if (argc == 2 && std::string(argv[1]) == "--dump-log") std::cout << output.str();
    for (const auto & failure : failures) std::cerr << failure << '\n';
    return failures.empty() ? 0 : 1;
}
