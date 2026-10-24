#pragma once

#include <algorithm>
#include <memory>
#include <vector>

namespace mstar::pi::detail {

// One canonical allocation has one physical ownership frontier. Vulkan's
// exclusive foreign-queue ownership requires even concurrent readers to be
// ordered. This provider adds that edge itself; callers still supply logical
// data dependencies. State::mutex serializes contribution and publication.
template <class Point>
class AllocationFrontier {
public:
    using Prefix = std::vector<std::shared_ptr<Point>>;

    void append_dependency(Prefix& prefix) const {
        if (auto prior = last_use_.lock()) {
            if (std::find(prefix.begin(), prefix.end(), prior) == prefix.end())
                prefix.push_back(std::move(prior));
        }
    }

    // Accepted operations are already retained by Pending before publication;
    // updating ownership after submission cannot allocate or throw.
    void publish(const std::shared_ptr<Point>& point) noexcept { last_use_ = point; }

private:
    // Expiry is safe because Pending retains every unsettled operation. Avoid
    // making an otherwise-unused allocation keep old fences alive forever.
    std::weak_ptr<Point> last_use_;
};

} // namespace mstar::pi::detail
