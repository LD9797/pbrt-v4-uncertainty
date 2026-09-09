#ifndef PBRT_NRC_NRC_H
#define PBRT_NRC_NRC_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace pbrt {
namespace nrc {

class NeuralRadianceCache {
  public:
    NeuralRadianceCache(uint32_t batchSize, uint32_t nInputDims,
                        uint32_t nOutputDims,
                        const std::string &configFile = "");
    ~NeuralRadianceCache();

    NeuralRadianceCache(const NeuralRadianceCache &) = delete;
    NeuralRadianceCache &operator=(const NeuralRadianceCache &) = delete;

    float Train(const float *dInputs, const float *dTargets);

    // One training step over the full batchSize. Returns the loss.
    float TrainN(const float *dInputs, const float *dTargets, uint32_t n);

    // Forward pass only. Writes nOutputDims*batchSize floats into dOutputs.
    void Inference(const float *dInputs, float *dOutputs);

    size_t NumParams() const;

    uint32_t BatchSize() const { return batchSize; }
    uint32_t NInputDims() const { return nInputDims; }
    uint32_t NOutputDims() const { return nOutputDims; }

    // Round n up to a valid tcnn batch size (CUDA-friendly granularity).
    static uint32_t RoundUpBatch(uint32_t n);

  private:
    struct Impl;
    Impl *impl;
    uint32_t batchSize;
    uint32_t nInputDims;
    uint32_t nOutputDims;
};

}  // namespace nrc
}  // namespace pbrt

#endif  // PBRT_NRC_NRC_H
