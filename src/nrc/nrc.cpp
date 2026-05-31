// Neural Radiance Cache — NRCRecordBuffer implementation.
// SPDX: Apache-2.0

#include <nrc/nrc.h>

namespace pbrt {

NRCRecordBuffer::NRCRecordBuffer(int capacity)
    : records_(capacity), capacity_(capacity) {}

int NRCRecordBuffer::Alloc() {
    int idx = count_.fetch_add(1, std::memory_order_relaxed);
    return idx < capacity_ ? idx : -1;
}

void NRCRecordBuffer::Reset() {
    count_.store(0, std::memory_order_relaxed);
}

}  // namespace pbrt
