// Neural Radiance Cache — CPU-side data structures.
// No CUDA headers are included here; this file is safe for pure-C++ TUs.
// SPDX: Apache-2.0

#pragma once

#include <atomic>
#include <vector>

namespace pbrt {

// ─────────────────────────────────────────────────────────────────────────────
// Feature layout (column-major in all GPU matrices: rows = features, cols = samples)
// ─────────────────────────────────────────────────────────────────────────────
//  f[ 0.. 2]  world-space position (x,y,z), normalised to [0,1] via scene AABB
//  f[ 3.. 5]  shading normal  (nx, ny, nz)
//  f[ 6.. 8]  outgoing direction wo  (dx, dy, dz)
//  f[ 9..11]  diffuse/hemispherical-directional reflectance  (R, G, B)
//  f[12]      roughness proxy (placeholder 0.5 until material query is added)
//  f[13..15]  padding zeros  (reserved for specular albedo)
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int NRC_INPUT_DIMS  = 16;
static constexpr int NRC_OUTPUT_DIMS = 3;    // RGB radiance

// One training sample.  'input' is filled when the surface vertex is visited;
// 'target' is filled after the full path terminates and the radiance is known.
struct NRCRecord {
    float input [NRC_INPUT_DIMS ];   // surface features
    float target[NRC_OUTPUT_DIMS];   // target incident RGB radiance
    bool  targetValid = false;       // set to true once target is written
};

// ─────────────────────────────────────────────────────────────────────────────
// Thread-safe CPU-side ring buffer.
// Records are *allocated* atomically (each thread gets its own slot), so
// writes to different slots never race.  Reset() must be called only when
// no threads are active.
// ─────────────────────────────────────────────────────────────────────────────
class NRCRecordBuffer {
  public:
    explicit NRCRecordBuffer(int capacity);

    // Atomically claim one slot.  Returns the index, or -1 when full.
    int Alloc();

    NRCRecord       &operator[](int i)       { return records_[i]; }
    const NRCRecord &operator[](int i) const { return records_[i]; }

    int Count()    const { return count_.load(std::memory_order_relaxed); }
    int Capacity() const { return capacity_; }

    // Zero the atomic counter so the buffer can be reused.  Call only when
    // no rendering threads are active.
    void Reset();

  private:
    std::vector<NRCRecord> records_;
    std::atomic<int>       count_{0};
    int                    capacity_;
};

}  // namespace pbrt
