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
    // dAux, if non-null, is a (2*nOutputDims)-per-sample matrix (same
    // per-sample stride convention as dTargets) forwarded verbatim to tcnn
    // as the training step's "data_pdf" argument. Vanilla losses
    // (RelativeL2, L2, ...) don't use it. It exists so pbrt-side code can
    // hand the SpectralRelativeL2 loss (see nrc_config.json) two
    // per-sample, per-channel auxiliary vectors it needs to turn the raw
    // network output into an actual radiance prediction and evaluate
    // Muller et al. 2021 Eq. 5 on it: channels [0, nOutputDims) are the
    // sample's spectral reflectance R = alpha+beta, and channels
    // [nOutputDims, 2*nOutputDims) are its CIE luminance weight -- this
    // class stays unaware of what the values actually mean; it just plumbs
    // them through.
    float TrainN(const float *dInputs, const float *dTargets, uint32_t n,
                const float *dAux = nullptr);


    // Forward pass only. Writes nOutputDims*batchSize floats into dOutputs.
    void Inference(const float *dInputs, float *dOutputs);

    // Forward pass over only the first n rows of dInputs/dOutputs (n must
    // be a multiple of tcnn's batch granularity, see RoundUpBatch). Lets
    // callers run inference over a compacted sub-batch instead of the full
    // batchSize when only a fraction of rows actually need a prediction.
    void InferenceN(const float *dInputs, float *dOutputs, uint32_t n);

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
