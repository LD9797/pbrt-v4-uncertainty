// Neural Radiance Cache — tiny-cuda-nn trainer implementation.
// Compiled by nvcc; plain C++ TUs only see nrc_trainer.h.
// SPDX: Apache-2.0

#include <nrc/nrc_trainer.h>

// ── tiny-cuda-nn ──────────────────────────────────────────────────────────────
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>

// ── CUDA ─────────────────────────────────────────────────────────────────────
#include <cuda_runtime.h>

// ── std ───────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
#define NRC_CUDA_CHECK(call)                                                   \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess)                                               \
            throw std::runtime_error(std::string("CUDA error in NRCTrainer: ")\
                                     + cudaGetErrorString(_err));              \
    } while (0)

// Round n down to the nearest multiple of align.
static int alignDown(int n, int align) { return (n / align) * align; }

namespace pbrt {

// ─────────────────────────────────────────────────────────────────────────────
// Pimpl struct — hides all tiny-cuda-nn types from the header
// ─────────────────────────────────────────────────────────────────────────────
struct NRCTrainer::Impl {
    tcnn::TrainableModel model;

    // Reusable GPU matrices; resized lazily as the batch changes.
    int batchCapacity = 0;
    std::unique_ptr<tcnn::GPUMatrix<float>> gpuInputs;
    std::unique_ptr<tcnn::GPUMatrix<float>> gpuTargets;
    std::unique_ptr<tcnn::GPUMatrix<float>> gpuOutputs;

    void resizeIfNeeded(int batchSize) {
        if (batchSize <= batchCapacity)
            return;
        gpuInputs  = std::make_unique<tcnn::GPUMatrix<float>>(NRC_INPUT_DIMS,  batchSize);
        gpuTargets = std::make_unique<tcnn::GPUMatrix<float>>(NRC_OUTPUT_DIMS, batchSize);
        gpuOutputs = std::make_unique<tcnn::GPUMatrix<float>>(NRC_OUTPUT_DIMS, batchSize);
        batchCapacity = batchSize;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Constructor — build the tiny-cuda-nn model
// ─────────────────────────────────────────────────────────────────────────────
NRCTrainer::NRCTrainer(int n_hidden_layers, int n_neurons, float learning_rate)
    : impl_(std::make_unique<Impl>()) {

    nlohmann::json config = {
        {"loss",      {{"otype", "L2"}}},
        {"optimizer", {
            {"otype",         "Adam"},
            {"learning_rate", learning_rate}
        }},
        {"encoding",  {{"otype", "Identity"}}},
        {"network",   {
            {"otype",            "FullyFusedMLP"},
            {"activation",       "ReLU"},
            {"output_activation","None"},
            {"n_neurons",        n_neurons},
            {"n_hidden_layers",  n_hidden_layers}
        }}
    };

    impl_->model = tcnn::create_from_config(NRC_INPUT_DIMS, NRC_OUTPUT_DIMS, config);

    std::cout << "[NRC] Network created — parameters: "
              << impl_->model.trainer->n_params() << "\n";
}

NRCTrainer::~NRCTrainer() = default;

// ─────────────────────────────────────────────────────────────────────────────
// Train
// ─────────────────────────────────────────────────────────────────────────────
float NRCTrainer::Train(const NRCRecordBuffer &buffer) {
    // Collect valid records into host-side flat arrays.
    // FullyFusedMLP requires the batch to be a multiple of 16; we use 256 for safety.
    const int align = 256;
    int total = buffer.Count();

    // Gather only valid records.
    std::vector<float> hInputs;
    std::vector<float> hTargets;
    hInputs .reserve(total * NRC_INPUT_DIMS );
    hTargets.reserve(total * NRC_OUTPUT_DIMS);

    for (int i = 0; i < total; ++i) {
        const NRCRecord &rec = buffer[i];
        if (!rec.targetValid) continue;
        for (int k = 0; k < NRC_INPUT_DIMS;  ++k) hInputs .push_back(rec.input [k]);
        for (int k = 0; k < NRC_OUTPUT_DIMS; ++k) hTargets.push_back(rec.target[k]);
    }

    int validCount = static_cast<int>(hTargets.size()) / NRC_OUTPUT_DIMS;
    if (validCount < kMinBatch) return -1.f;

    // Truncate to aligned batch size (tiny-cuda-nn GPUMatrix must match exactly).
    int batchSize = alignDown(validCount, align);

    impl_->resizeIfNeeded(batchSize);
    auto &inputs  = *impl_->gpuInputs;
    auto &targets = *impl_->gpuTargets;

    // ── host arrays are row-major [sample × feature]; GPUMatrix is column-major
    //    [feature × sample].  Transpose on upload. ──────────────────────────────
    std::vector<float> colInputs (NRC_INPUT_DIMS  * batchSize);
    std::vector<float> colTargets(NRC_OUTPUT_DIMS * batchSize);

    for (int col = 0; col < batchSize; ++col) {
        for (int row = 0; row < NRC_INPUT_DIMS; ++row)
            colInputs[col * NRC_INPUT_DIMS + row] =
                hInputs[col * NRC_INPUT_DIMS + row];
        for (int row = 0; row < NRC_OUTPUT_DIMS; ++row)
            colTargets[col * NRC_OUTPUT_DIMS + row] =
                hTargets[col * NRC_OUTPUT_DIMS + row];
    }

    NRC_CUDA_CHECK(cudaMemcpy(inputs .data(),
                              colInputs .data(),
                              colInputs .size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    NRC_CUDA_CHECK(cudaMemcpy(targets.data(),
                              colTargets.data(),
                              colTargets.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

    // ── Training step ─────────────────────────────────────────────────────────
    auto ctx  = impl_->model.trainer->training_step(inputs, targets);
    float loss = impl_->model.trainer->loss(*ctx);
    NRC_CUDA_CHECK(cudaDeviceSynchronize());

    ++trainSteps_;
    return loss;
}

// ─────────────────────────────────────────────────────────────────────────────
// Inference
// ─────────────────────────────────────────────────────────────────────────────
void NRCTrainer::Inference(const float *inputs, int n, float *outputs) const {
    if (n <= 0) return;

    const int align     = 16;   // FullyFusedMLP alignment for inference
    int       batchSize = alignDown(n, align);
    if (batchSize == 0) return;

    // We re-use the resize logic; cast away const via Impl pointer.
    impl_->resizeIfNeeded(batchSize);
    auto &gpuIn  = *impl_->gpuInputs;
    auto &gpuOut = *impl_->gpuOutputs;

    // Host → GPU  (transpose to column-major)
    std::vector<float> colInputs(NRC_INPUT_DIMS * batchSize);
    for (int col = 0; col < batchSize; ++col)
        for (int row = 0; row < NRC_INPUT_DIMS; ++row)
            colInputs[col * NRC_INPUT_DIMS + row] =
                inputs[col * NRC_INPUT_DIMS + row];

    NRC_CUDA_CHECK(cudaMemcpy(gpuIn.data(),
                              colInputs.data(),
                              colInputs.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

    impl_->model.network->inference(gpuIn, gpuOut);
    NRC_CUDA_CHECK(cudaDeviceSynchronize());

    // GPU → host  (transpose back to row-major)
    std::vector<float> colOutputs(NRC_OUTPUT_DIMS * batchSize);
    NRC_CUDA_CHECK(cudaMemcpy(colOutputs.data(),
                              gpuOut.data(),
                              colOutputs.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));

    for (int col = 0; col < batchSize; ++col)
        for (int row = 0; row < NRC_OUTPUT_DIMS; ++row)
            outputs[col * NRC_OUTPUT_DIMS + row] =
                colOutputs[col * NRC_OUTPUT_DIMS + row];
}

}  // namespace pbrt
