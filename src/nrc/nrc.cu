#include <nrc/nrc.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace pbrt
{
    namespace nrc
    {

        struct NeuralRadianceCache::Impl
        {
            tcnn::TrainableModel model;
            cudaStream_t stream = nullptr;
            float lastLoss = 0.f;

            Impl(uint32_t nIn, uint32_t nOut)
                : model(tcnn::create_from_config(
                      nIn, nOut,
                      {{"loss", {{"otype", "L2"}}},
                       {"optimizer", {{"otype", "Adam"}, {"learning_rate", 1e-3}}},
                       {"encoding", {{"otype", "Identity"}}},
                       {"network",
                        {{"otype", "FullyFusedMLP"},
                         {"activation", "ReLU"},
                         {"output_activation", "None"},
                         {"n_neurons", 64},
                         {"n_hidden_layers", 2}}}})) {}
        };

        NeuralRadianceCache::NeuralRadianceCache(uint32_t batchSize_, uint32_t nIn,
                                                 uint32_t nOut)
            : batchSize(batchSize_), nInputDims(nIn), nOutputDims(nOut)
        {
            if (batchSize % tcnn::batch_size_granularity != 0)
            {
                throw std::runtime_error(
                    "NeuralRadianceCache: batchSize must be a multiple of " +
                    std::to_string(tcnn::batch_size_granularity));
            }
            impl = new Impl(nIn, nOut);
        }

        NeuralRadianceCache::~NeuralRadianceCache()
        {
            delete impl;
        }

        uint32_t NeuralRadianceCache::RoundUpBatch(uint32_t n)
        {
            return tcnn::next_multiple(n, (uint32_t)tcnn::batch_size_granularity);
        }

        size_t NeuralRadianceCache::NumParams() const
        {
            return impl->model.trainer->n_params();
        }

    } 
} 
