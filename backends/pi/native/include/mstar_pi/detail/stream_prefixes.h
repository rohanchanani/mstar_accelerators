#pragma once

#include "mstar_pi/runtime.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <vector>

namespace mstar::pi::detail {

// The backend's logical stream bookkeeping, independent of the native fence
// representation. A snapshot is a value. Waiting joins both prefixes; only a
// submitted operation that consumes that join may replace it with one fence.
// This is internal to the Torch backend, not another runtime scheduler.
template <class Point>
class StreamPrefixes {
public:
    using Prefix = std::vector<std::shared_ptr<Point>>;

    StreamId create() {
        if (next_ == std::numeric_limits<StreamId>::max()) throw Rejected("stream identifier space exhausted");
        const auto id = next_;
        values_.emplace(id, Prefix{});
        ++next_;
        return id;
    }

    Prefix& at(StreamId id) {
        const auto it = values_.find(id);
        if (it == values_.end()) throw Rejected("unknown Pi logical stream");
        return it->second;
    }

    Prefix snapshot(StreamId id) { return at(id); }

    void join(StreamId id, const Prefix& dependency) {
        auto& destination = at(id);
        Prefix joined = destination;
        for (const auto& point : dependency) {
            if (std::find(joined.begin(), joined.end(), point) == joined.end()) joined.push_back(point);
        }
        destination.swap(joined);
    }

    Prefix snapshot_all() const {
        Prefix result;
        for (const auto& item : values_)
            result.insert(result.end(), item.second.begin(), item.second.end());
        return result;
    }

    void clear() { values_.clear(); }

private:
    std::map<StreamId, Prefix> values_{{0, {}}};
    StreamId next_ = 1;
};

} // namespace mstar::pi::detail
