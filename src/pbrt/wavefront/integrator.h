// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_WAVEFRONT_INTEGRATOR_H
#define PBRT_WAVEFRONT_INTEGRATOR_H

#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/filter.h>
#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/base/sampler.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/util.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/options.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/pstd.h>
#include <pbrt/wavefront/workitems.h>
#include <pbrt/wavefront/workqueue.h>

#include <cstdint>
#include <string>

namespace pbrt {

#ifdef PBRT_BUILD_NRC
namespace nrc {
class NeuralRadianceCache;
}
#endif

class BasicScene;
class GUI;

// WavefrontAggregate Definition
class WavefrontAggregate {
  public:
    // WavefrontAggregate Interface
    virtual ~WavefrontAggregate() = default;

    virtual Bounds3f Bounds() const = 0;

    virtual void IntersectClosest(int maxRays, const RayQueue *rayQ,
                                  EscapedRayQueue *escapedRayQ,
                                  HitAreaLightQueue *hitAreaLightQ,
                                  MaterialEvalQueue *basicMtlQ,
                                  MaterialEvalQueue *universalMtlQ,
                                  MediumSampleQueue *mediumSampleQ,
                                  RayQueue *nextRayQ) const = 0;

    virtual void IntersectShadow(int maxRays, ShadowRayQueue *shadowRayQueue,
                                 SOA<PixelSampleState> *pixelSampleState) const = 0;
    virtual void IntersectShadowTr(int maxRays, ShadowRayQueue *shadowRayQueue,
                                   SOA<PixelSampleState> *pixelSampleState) const = 0;

    virtual void IntersectOneRandom(
        int maxRays, SubsurfaceScatterQueue *subsurfaceScatterQueue) const = 0;
};

// WavefrontPathIntegrator Definition
class WavefrontPathIntegrator {
  public:
    // WavefrontPathIntegrator Public Methods
    Float Render();

    void GenerateCameraRays(int y0, Transform movingFromcamera, int sampleIndex);
    template <typename Sampler>
    void GenerateCameraRays(int y0, Transform movingFromCamera, int sampleIndex);

    void GenerateRaySamples(int wavefrontDepth, int sampleIndex);
    template <typename Sampler>
    void GenerateRaySamples(int wavefrontDepth, int sampleIndex);

    void TraceShadowRays(int wavefrontDepth);
    void SampleMediumInteraction(int wavefrontDepth);
    template <typename PhaseFunction>
    void SampleMediumScattering(int wavefrontDepth);
    void SampleSubsurface(int wavefrontDepth);

    void HandleEscapedRays();
    void HandleEmissiveIntersection();

    void EvaluateMaterialsAndBSDFs(int wavefrontDepth, Transform movingFromCamera);
    template <typename ConcreteMaterial>
    void EvaluateMaterialAndBSDF(int wavefrontDepth, Transform movingFromCamera);
    template <typename ConcreteMaterial, typename TextureEvaluator>
    void EvaluateMaterialAndBSDF(MaterialEvalQueue *evalQueue, Transform movingFromCamera,
                                 int wavefrontDepth);

    void UpdateFilm();

    WavefrontPathIntegrator(pstd::pmr::memory_resource *memoryResource,
                            BasicScene &scene);

    template <typename F>
    void ParallelFor(const char *description, int nItems, F &&func) {
        if (Options->useGPU)
#ifdef PBRT_BUILD_GPU_RENDERER
            GPUParallelFor(description, nItems, func);
#else
            LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif
        else
            pbrt::ParallelFor(0, nItems, func);
    }

    template <typename F>
    void Do(const char *description, F &&func) {
        if (Options->useGPU)
#ifdef PBRT_BUILD_GPU_RENDERER
            GPUParallelFor(description, 1, [=] PBRT_GPU(int) mutable { func(); });
#else
            LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif
        else
            func();
    }

    RayQueue *CurrentRayQueue(int wavefrontDepth) {
        return rayQueues[wavefrontDepth & 1];
    }
    RayQueue *NextRayQueue(int wavefrontDepth) {
        return rayQueues[(wavefrontDepth + 1) & 1];
    }

#ifdef PBRT_BUILD_GPU_RENDERER
    void PrefetchGPUAllocations();
#endif  // PBRT_BUILD_GPU_RENDERER

    // --display-server methods
    void StartDisplayThread();
    void UpdateDisplayRGBFromFilm(Bounds2i pixelBounds);
    void StopDisplayThread();

    // --interactive support
    void UpdateFramebufferFromFilm(Bounds2i pixelBounds, Float exposure, RGB *rgb);

    // WavefrontPathIntegrator Member Variables
    bool initializeVisibleSurface;
    bool haveSubsurface;
    bool haveMedia;
    pstd::array<bool, Material::NumTags()> haveBasicEvalMaterial;
    pstd::array<bool, Material::NumTags()> haveUniversalEvalMaterial;

    struct Stats {
        Stats(int maxDepth, Allocator alloc);

        std::string Print() const;

        // Note: not atomics: tid 0 always updates them for everyone...
        uint64_t cameraRays = 0;
        pstd::vector<uint64_t> indirectRays, shadowRays;
    };
    Stats *stats;

    pstd::pmr::memory_resource *memoryResource;

    Filter filter;
    Film film;
    Sampler sampler;
    Camera camera;
    pstd::vector<Light> *infiniteLights;
    LightSampler lightSampler;

    int maxDepth, samplesPerPixel;
    bool regularize;

    int scanlinesPerPass, maxQueueSize;

    SOA<PixelSampleState> pixelSampleState;

    RayQueue *rayQueues[2];

    WavefrontAggregate *aggregate = nullptr;

    MediumSampleQueue *mediumSampleQueue = nullptr;
    MediumScatterQueue *mediumScatterQueue = nullptr;

    EscapedRayQueue *escapedRayQueue = nullptr;

    HitAreaLightQueue *hitAreaLightQueue = nullptr;

    MaterialEvalQueue *basicEvalMaterialQueue = nullptr;
    MaterialEvalQueue *universalEvalMaterialQueue = nullptr;

    ShadowRayQueue *shadowRayQueue = nullptr;

    GetBSSRDFAndProbeRayQueue *bssrdfEvalQueue = nullptr;
    SubsurfaceScatterQueue *subsurfaceScatterQueue = nullptr;

    RGB *displayRGB = nullptr, *displayRGBHost = nullptr;
    std::atomic<bool> *exitCopyThread;
    std::thread *copyThread;

#ifdef PBRT_BUILD_NRC
    // ---------------- Neural Radiance Cache (milestone 2) ----------------
    // Buffers are CUDA-managed and indexed by pixelIndex in [0, maxQueueSize).
    // Layouts are column-major to match tcnn::GPUMatrix:
    //   nrcInputs: 49 raw floats per slot, matching Muller et al. 2021's NRC
    //   input layer 1:1. Position is frequency-encoded here in PBRT (sin-only,
    //   12 bands/axis, per the paper -- tcnn's own Frequency encoding also emits
    //   cosine, which the paper omits) and passed through tcnn as raw Identity
    //   dims; the remaining fields are still encoded by nrc_config.json:
    //     0-35: position, normalized to [0,1] via scene bounds, then encoded as
    //           sin((1<<d) * x) for d in [0,12) per axis (-> Identity = 36)
    //     36-37: outgoing direction, spherical (theta,phi in [0,1]; -> OneBlob(4) = 8)
    //     38-39: shading normal, spherical (theta,phi in [0,1]; -> OneBlob(4) = 8)
    //     40:   roughness, transformed 1-exp(-r) (-> OneBlob(4) = 4)
    //     41-43: diffuse albedo RGB (bsdf.rho via ToOutputRGB; raw, Identity)
    //     44-46: specular reflectance F0 RGB (Dielectric/Conductor Fresnel at
    //            normal incidence; 0 for other types; raw, Identity)
    //     47-48: padding, constant 1 (paper pads to 64 for tcnn tile alignment)
    //   Total encoded width: 36+8+8+4+3+3+2 = 64
    //   nrcTargets: 3 floats per slot (signed-log-encoded RGB radiance)
    //   nrcValid:   1 byte per slot, set at depth==0 first-hit
    //   nrcTrainingPath: 1 byte per slot, set by GenerateCameraRays before the
    //     first hit is even traced. 
    static constexpr uint32_t kNRCInputDims = 49;
    static constexpr uint32_t kNRCOutputDims = 3;
    uint32_t nrcBatchSize = 0;  // = NeuralRadianceCache::RoundUpBatch(maxQueueSize)
    Bounds3f nrcSceneBounds;        // set from aggregate->Bounds() at init; used to normalize pos to [-1,1]
    float *nrcInputs = nullptr;
    float *nrcTargets = nullptr;
    uint8_t *nrcValid = nullptr;
    uint8_t *nrcTrainingPath = nullptr;  // 1 = this path was selected to generate a training record this pass
    bool nrcCaptureAll = false;  // true during the final inference sweep: capture every path, ignore selection
    float *nrcCompactInputs  = nullptr;  // valid-only training inputs, compacted each pass
    float *nrcCompactTargets = nullptr;  // valid-only training targets, compacted each pass
    float *nrcInferenceOutputs = nullptr;  // scratch for per-step inference
    // Persistent per-pixel predicted RGB image (sized to film resolution).
    // Populated by per-sample inference passes; written to EXR at end of Render().
    float *nrcPredictedRGB = nullptr;
    Point2i nrcResolution = {0, 0};
    nrc::NeuralRadianceCache *nrcCache = nullptr;
    int nrcSampleCounter = 0;
    float nrcLastLoss = 0.f;

    void NRCResetSampleBuffers();
    void NRCCaptureFinalRadiance();
    void NRCTrainAndInferStep();
    void NRCDumpPredictedImage(const std::string &filename);
#endif  // PBRT_BUILD_NRC
};

}  // namespace pbrt

#endif  // PBRT_WAVEFRONT_INTEGRATOR_H
