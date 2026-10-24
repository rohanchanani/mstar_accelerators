#include "mstar_pi/detail/stream_prefixes.h"
#include "mstar_pi/detail/allocation_frontier.h"

#include <iostream>
#include <memory>
#include <stdexcept>

static void require(bool value) {
    if (!value) throw std::runtime_error("stream-prefix contract check failed");
}

struct Point { int id; bool complete = false; };
using Streams = mstar::pi::detail::StreamPrefixes<Point>;

static bool ready(const Streams::Prefix& prefix) {
    for (const auto& point : prefix) if (!point->complete) return false;
    return true;
}

int main() {
    Streams streams;
    const auto a_stream = streams.create();
    const auto b_stream = streams.create();
    auto a = std::make_shared<Point>(Point{1});
    auto b = std::make_shared<Point>(Point{2});
    auto c = std::make_shared<Point>(Point{3});
    streams.at(a_stream) = {a};
    streams.at(b_stream) = {b};
    const auto a_event = streams.snapshot(a_stream);
    const auto old_b_event = streams.snapshot(b_stream);

    // The destination's existing work is a prerequisite too. Completing A
    // cannot make the joined stream ready while its original B is outstanding.
    streams.join(b_stream, a_event);
    const auto joined_event = streams.snapshot(b_stream);
    require(joined_event.size() == 2);
    a->complete = true;
    require(ready(a_event));
    require(!ready(joined_event));
    b->complete = true;
    require(ready(joined_event));

    // A later submission replaces the stream tail after consuming the join;
    // previously recorded snapshots retain their original meaning.
    streams.at(b_stream) = {c};
    require(ready(old_b_event));
    require(ready(joined_event));
    require(!ready(streams.snapshot(b_stream)));
    streams.join(b_stream, a_event);
    streams.join(b_stream, a_event);
    require(streams.snapshot(b_stream).size() == 2); // no duplicate prerequisites
    c->complete = true;
    require(ready(streams.snapshot(b_stream)));

    // Empty events do not erase prior work. Snapshots own their points even
    // after stream bookkeeping and the caller's original handles disappear.
    c->complete = false;
    streams.join(b_stream, {});
    const auto retained = streams.snapshot(b_stream);
    std::weak_ptr<Point> weak_c = c;
    streams.clear();
    a.reset(); b.reset(); c.reset();
    require(!weak_c.expired());
    require(!ready(retained));
    weak_c.lock()->complete = true;
    require(ready(retained));

    bool rejected = false;
    try { streams.join(999, {}); }
    catch (const mstar::pi::Rejected&) { rejected = true; }
    require(rejected);

    // Independent logical readers still need an exclusive foreign-allocation
    // ownership edge. It must augment, never replace, the consumer's stream
    // prefix; another allocation must not introduce a spurious global edge.
    mstar::pi::detail::AllocationFrontier<Point> shared_allocation;
    mstar::pi::detail::AllocationFrontier<Point> separate_allocation;
    auto first_reader = std::make_shared<Point>(Point{4});
    auto logical_input = std::make_shared<Point>(Point{5});
    auto second_reader = std::make_shared<Point>(Point{6});
    shared_allocation.publish(first_reader);
    Streams::Prefix second_inputs{logical_input};
    shared_allocation.append_dependency(second_inputs);
    shared_allocation.append_dependency(second_inputs);
    require(second_inputs.size() == 2);
    logical_input->complete = true;
    require(!ready(second_inputs));
    first_reader->complete = true;
    require(ready(second_inputs));
    separate_allocation.append_dependency(second_inputs);
    require(second_inputs.size() == 2);
    shared_allocation.publish(second_reader);
    Streams::Prefix third_inputs;
    shared_allocation.append_dependency(third_inputs);
    require(third_inputs.size() == 1 && third_inputs[0] == second_reader);
    require(ready(second_inputs)); // publication cannot mutate prior prefixes
    second_reader->complete = true;
    second_reader.reset();
    third_inputs.clear(); // no unsettled Pending would release its point
    shared_allocation.append_dependency(third_inputs);
    require(third_inputs.empty());
    std::cout << "PASS: stream fan-in, immutable event snapshots, allocation ownership frontier, invalid-stream rejection\n";
}
