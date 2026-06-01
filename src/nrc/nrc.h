#ifndef PBRT_NRC_NRC_H
#define PBRT_NRC_NRC_H

#include <cstddef>
#include <cstdint>

namespace pbrt
{
    namespace nrc
    {

        class NeuralRadianceCache
        {
        public:
            NeuralRadianceCache(uint32_t batchSize, uint32_t nInputDims, uint32_t nOutputDims);
            ~NeuralRadianceCache();

            static uint32_t RoundUpBatch(uint32_t n);

            size_t NumParams() const;

        private:
            struct Impl;
            Impl *impl;
            uint32_t batchSize;
            uint32_t nInputDims;
            uint32_t nOutputDims;
        };

    }
}

#endif
