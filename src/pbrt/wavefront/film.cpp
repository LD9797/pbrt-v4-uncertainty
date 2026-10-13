// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/film.h>
#include <pbrt/wavefront/integrator.h>

namespace pbrt {

// WavefrontPathIntegrator Film Methods
void WavefrontPathIntegrator::UpdateFilm() {
    ParallelFor(
        "Update film", maxQueueSize, PBRT_CPU_GPU_LAMBDA(int pixelIndex) {
            // Check pixel against film bounds
            Point2i pPixel = pixelSampleState.pPixel[pixelIndex];
            if (!InsideExclusive(pPixel, film.PixelBounds()))
                return;

            // Compute final weighted radiance value
            SampledSpectrum Lw = SampledSpectrum(pixelSampleState.L[pixelIndex]) *
                                 pixelSampleState.cameraRayWeight[pixelIndex];

            PBRT_DBG("Adding Lw %f %f %f %f at pixel (%d, %d)\n", Lw[0], Lw[1], Lw[2],
                     Lw[3], pPixel.x, pPixel.y);
            // Provide sample radiance value to film
            SampledWavelengths lambda = pixelSampleState.lambda[pixelIndex];
            Float filterWeight = pixelSampleState.filterWeight[pixelIndex];

#ifdef PBRT_BUILD_NRC
            // Training-path supervision no longer comes from here. Every
            // training path that reaches its render-query vertex now gets
            // an explicit "training suffix" (see surfscatter.cpp's
            // nrcSuffixActive tracking, integrator.cpp's
            // NRCTrainingSuffixFinish()): one backward-propagated record per
            // suffix vertex, built purely by forward multiplication of a
            // local (reset-to-1) throughput, never by dividing the real
            // path's beta. nrcValid is therefore never set for training
            // paths anymore; it's reserved for non-training (render-query)
            // paths' inference-only use in NRCInferenceForRenderPaths().
#endif

            if (initializeVisibleSurface) {
                // Call _Film::AddSample()_ with _VisibleSurface_ for pixel sample
                VisibleSurface visibleSurface =
                    pixelSampleState.visibleSurface[pixelIndex];
                film.AddSample(pPixel, Lw, lambda, &visibleSurface, filterWeight);

            } else
                film.AddSample(pPixel, Lw, lambda, nullptr, filterWeight);
        });
}

}  // namespace pbrt
