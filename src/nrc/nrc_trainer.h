// Neural Radiance Cache — GPU trainer interface.
// The implementation lives in nrc_trainer.cu so that plain C++ TUs only see
// this header and never pull in CUDA-specific types.
// SPDX: Apache-2.0

#pragma once

#include <nrc/nrc.h>
#include <memory>

namespace pbrt {

// ─────────────────────────────────────────────────────────────────────────────
// NRCTrainer
//
// Wraps a tiny-cuda-nn FullyFusedMLP with the Adam optimiser and L2 loss.
// Network architecture mirrors milestone1:
//   input   = NRC_INPUT_DIMS  (16)
//   output  = NRC_OUTPUT_DIMS (3, linear)
//   hidden  = n_neurons neurons, n_hidden_layers layers, ReLU activation
// ─────────────────────────────────────────────────────────────────────────────
class NRCTrainer {
  public:
    // Construct the network.  This allocates GPU memory and JIT-compiles the
    // CUDA kernels on first use.
    explicit NRCTrainer(int n_hidden_layers = 4,
                        int n_neurons       = 64,
                        float learning_rate = 1e-3f);
    ~NRCTrainer();

    // ── Training ──────────────────────────────────────────────────────────────
    // Upload all valid records from 'buffer' to the GPU, run one Adam step, and
    // return the mean L2 loss.  Batches with fewer than kMinBatch valid records
    // are skipped (returns -1.f).
    float Train(const NRCRecordBuffer &buffer);

    // ── Inference ─────────────────────────────────────────────────────────────
    // Run the network on 'n' input rows stored in host memory.
    //   inputs  : [n × NRC_INPUT_DIMS ] row-major floats (host pointer)
    //   outputs : [n × NRC_OUTPUT_DIMS] row-major floats (host pointer, caller allocs)
    void Inference(const float *inputs, int n, float *outputs) const;

    int TrainSteps() const { return trainSteps_; }

    // Minimum number of valid records needed before we attempt a training step.
    static constexpr int kMinBatch = 256;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int trainSteps_ = 0;
};

}  // namespace pbrt
