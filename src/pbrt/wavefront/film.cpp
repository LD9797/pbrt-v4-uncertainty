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
            // Pair this sample's continuation-only radiance with its
            // captured query-vertex input row. The path kept tracing past
            // its query vertex (training paths never terminate early -- see
            // surfscatter.cpp), so pixelSampleState.L now also holds
            // whatever was accumulated after that vertex. Subtract off the
            // nrcSnapshotL taken right before the vertex's own shading to
            // isolate that continuation, then divide by the throughput that
            // reached the vertex (nrcSnapshotBeta) so the target is the
            // vertex's local outgoing radiance, independent of path-specific
            // throughput -- matching what NRCInferenceForRenderPaths()
            // substitutes at render time (nrcSnapshotL + nrcSnapshotBeta *
            // predicted).
            if (nrcTargets != nullptr && nrcValid != nullptr &&
                nrcValid[pixelIndex]) {
                SampledSpectrum Lraw = pixelSampleState.L[pixelIndex];
                SampledSpectrum Lsnapshot, betaSnapshot;
                for (int c = 0; c < NSpectrumSamples; ++c) {
                    Lsnapshot[c] = nrcSnapshotL[size_t(pixelIndex) * NSpectrumSamples + c];
                    betaSnapshot[c] = nrcSnapshotBeta[size_t(pixelIndex) * NSpectrumSamples + c];
                }
                SampledSpectrum Lo = SafeDiv(Lraw - Lsnapshot, betaSnapshot);
                RGB rgb = film.ToOutputRGB(Lo, lambda);
                float *t = nrcTargets +
                           size_t(pixelIndex) * kNRCOutputDims;
                t[0] = float(rgb.r);
                t[1] = float(rgb.g);
                t[2] = float(rgb.b);
            }
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
