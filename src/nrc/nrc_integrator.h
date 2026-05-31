// Neural Radiance Cache — NRC-augmented path integrator.
//
// Integration roadmap (all controlled by constructor parameters / scene file options):
//   Step 3  — collect one NRCRecord per path (Li() with record hooks)
//   Step 4  — train tiny-cuda-nn after the render wave (Render() override)
//   Step 5  — print loss and target/prediction examples
//   Step 6  — write a debug EXR showing predicted vs ground-truth radiance
//   Step 7  — terminate paths early using NRC predictions
//
// Scene-file usage:
//   Integrator "nrcpath"
//       "integer maxdepth"    [5]
//       "integer recorddepth" [2]          # bounce at which to record the cache point
//       "bool    useprediction" [false]    # Step 7: terminate paths at record depth
//       "bool    debugimage"    [false]    # Step 6: write debug_nrc.exr
//       "integer nhiddenlayers" [4]
//       "integer nneurons"      [64]
// SPDX: Apache-2.0

#pragma once

#include <pbrt/cpu/integrators.h>
#include <nrc/nrc.h>
#include <nrc/nrc_trainer.h>

#include <memory>
#include <string>
#include <vector>

namespace pbrt {

// ─────────────────────────────────────────────────────────────────────────────
class NRCPathIntegrator : public RayIntegrator {
  public:
    NRCPathIntegrator(int maxDepth,
                      Camera camera,
                      Sampler sampler,
                      Primitive aggregate,
                      std::vector<Light> lights,
                      const std::string &lightSampleStrategy = "bvh",
                      bool regularize       = false,
                      int  recordDepth      = 2,
                      bool usePrediction    = false,
                      bool writeDebugImage  = false,
                      int  nHiddenLayers    = 4,
                      int  nNeurons         = 64);

    static std::unique_ptr<NRCPathIntegrator> Create(
        const ParameterDictionary &parameters,
        Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights,
        const FileLoc *loc);

    std::string ToString() const override;

    // ── Override Render to add training + debug passes ───────────────────────
    void Render() override;

    // ── Li() — path tracing with NRC record collection ───────────────────────
    SampledSpectrum Li(RayDifferential ray,
                       SampledWavelengths &lambda,
                       Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const override;

  private:
    // Direct-lighting helper (mirrors PathIntegrator::SampleLd).
    SampledSpectrum SampleLd(const SurfaceInteraction &intr,
                             const BSDF *bsdf,
                             SampledWavelengths &lambda,
                             Sampler sampler) const;

    // Encode a surface vertex into NRC_INPUT_DIMS features.
    void FillInputFeatures(float *f16,
                           const SurfaceInteraction &isect,
                           const BSDF &bsdf,
                           const SampledWavelengths &lambda) const;

    // Write debug_nrc.exr with NRC-only predictions (step 6).
    void WriteDebugImage();

    // Print a few target / prediction pairs (step 5).
    void PrintExamples(float loss) const;

    // ── Config ───────────────────────────────────────────────────────────────
    int         maxDepth_;
    int         recordDepth_;       // which bounce index triggers recording
    bool        regularize_;
    bool        usePrediction_;     // Step 7: replace continuation with NRC
    bool        writeDebugImage_;   // Step 6: write debug EXR after training
    LightSampler lightSampler_;
    Bounds3f    sceneBounds_;

    // ── NRC state (mutable so Li() — which is const — can write records) ─────
    mutable std::unique_ptr<NRCRecordBuffer> nrcBuffer_;
    std::unique_ptr<NRCTrainer>              trainer_;
};

}  // namespace pbrt
