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
    //   nrcValid:   1 byte per slot, set only when a training record was
    //     actually written into nrcInputs/nrcTargets (i.e. this path both
    //     reached its NRC query vertex AND was selected as a training path)
    //   nrcReachedQueryVertex: 1 byte per slot, set as soon as ANY path (not
    //     just training paths) crosses the area-spread threshold below --
    //     stops that path's spread tracking from running again at later
    //     vertices, independent of whether a training record gets written
    //   nrcTrainingPath: 1 byte per slot, set by GenerateCameraRays before the
    //     first hit is even traced.
    //   nrcRenderQuery: 1 byte per slot, set when a *non-training* path's
    //     query vertex was chosen by the area-spread heuristic itself or by
    //     max-depth truncation (NOT by Russian roulette or an invalid BSDF
    //     sample, since those are natural dead ends with no expensive
    //     continuation left to approximate). Such paths skip their indirect
    //     bounce and NEE at that vertex; NRCInferenceForRenderPaths() queries
    //     the cache for all of them once per scanline pass (after the depth
    //     loop, before UpdateFilm()) and adds beta * predicted radiance into
    //     L, replacing the skipped continuation with Muller et al.'s cache
    //     estimate.
    //   nrcSnapshotBeta/nrcSnapshotL: NSpectrumSamples floats per slot,
    //     captured the moment a path's query vertex is found (whichever of
    //     the four ways above), regardless of training/render status.
    //     nrcSnapshotBeta is the path throughput arriving at that vertex
    //     (before any of its own scattering); nrcSnapshotL is pixelSampleState
    //     .L just before that vertex's own NEE runs. UpdateFilm() uses
    //     nrcSnapshotL to turn the full-path Lw into a continuation-only
    //     training target: Lw - nrcSnapshotL, matching what
    //     NRCInferenceForRenderPaths() adds directly at render time
    //     (nrcSnapshotL + predicted). nrcSnapshotBeta is captured but
    //     deliberately NOT divided into the target -- doing so blows up
    //     whenever the throughput is small, producing huge targets and an
    //     over-bright image.
    static constexpr uint32_t kNRCInputDims = 49;
    static constexpr uint32_t kNRCOutputDims = 3;

    // Muller et al. 2021 Sec. 3.4 "Path Termination": all paths are
    // terminated according to the area-spread heuristic below, which picks
    // the query vertex dynamically per path (rather than always the first
    // hit). It is the *primary* rule, but not the only way a path can stop:
    // if max-depth truncation or Russian roulette ends the path first, the
    // last vertex actually reached becomes the query vertex by necessity.
    //   a(x1..xi) = (sum_{k=2}^{i} sqrt(||x_{k-1}-x_k||^2 /
    //                (p(wk | x_{k-1}) * |cos(theta_k)|)))^2         (Eq. 3)
    //   a0 = ||x0-x1||^2 / (4*Pi*|cos(theta1)|)                     (Eq. 4)
    // Terminate (this vertex becomes the query vertex) once a > c * a0.
    static constexpr float kNRCSpreadC = 0.01f;
    float *nrcPathSpreadAccum = nullptr;  // running sum of sqrt(...) terms in Eq. 3, reset per camera sample
    float *nrcPathA0 = nullptr;           // baseline footprint at the primary vertex, Eq. 4
    Point3f *nrcPathPrevP = nullptr;      // position of the previous path vertex; primed with the camera position
    float *nrcPathPrevPdf = nullptr;      // solid-angle pdf used to sample the direction that reached the current vertex
    uint32_t nrcBatchSize = 0;  // = NeuralRadianceCache::RoundUpBatch(maxQueueSize)
    Bounds3f nrcSceneBounds;        // set from aggregate->Bounds() at init; used to normalize pos to [-1,1]
    float *nrcInputs = nullptr;
    float *nrcTargets = nullptr;
    uint8_t *nrcValid = nullptr;
    uint8_t *nrcReachedQueryVertex = nullptr;  // 1 = this path has already found its NRC query vertex
    uint8_t *nrcTrainingPath = nullptr;  // 1 = this path was selected to generate a training record this pass
    bool nrcCaptureAll = false;  // true during the final inference sweep: capture every path, ignore selection
    float *nrcCompactInputs  = nullptr;  // valid-only training inputs, compacted each pass
    float *nrcCompactTargets = nullptr;  // valid-only training targets, compacted each pass
    float *nrcInferenceOutputs = nullptr;  // scratch for per-step inference
    uint8_t *nrcRenderQuery = nullptr;   // 1 = non-training path terminated at its query vertex this pass; needs cache substitution
    float *nrcSnapshotBeta = nullptr;    // NSpectrumSamples floats/slot: path throughput arriving at the query vertex
    float *nrcSnapshotL = nullptr;       // NSpectrumSamples floats/slot: L accumulated strictly before the query vertex's own shading
    // Persistent per-pixel predicted RGB image (sized to film resolution).
    // Populated by per-sample inference passes; written to EXR at end of Render().
    float *nrcPredictedRGB = nullptr;
    Point2i nrcResolution = {0, 0};
    nrc::NeuralRadianceCache *nrcCache = nullptr;
    int nrcSampleCounter = 0;
    float nrcLastLoss = 0.f;
    // False until nrcSampleCounter reaches Options->nrcWarmupSamples; while
    // false, nrcTerminateAndSubstitute (surfscatter.cpp) never fires, so
    // every path traces to completion for real and only trains the cache --
    // avoids baking an untrained network's predictions permanently into the
    // progressively-accumulated film.
    bool nrcWarmedUp = false;

    // ---- Training-suffix self-training (Muller et al. Sec. 3 / Fig. 4) ----
    // A training path never terminates at its render-query vertex (it never
    // did). What's new here: the continuation past that vertex is now
    // explicitly tracked as a short "training suffix". The SAME area-spread
    // heuristic (Eq. 3/4) is re-applied independently to the suffix, treating
    // the render-query vertex as a new "primary vertex" for that second test,
    // with a hard cap of kNRCMaxSuffixLen extra vertices as a GPU-memory
    // safety net. When the suffix ends via its own heuristic/cap, a bootstrap
    // NRC query supplies the missing continuation; when it ends naturally
    // (invalid BSDF sample / Russian roulette), no bootstrap is needed. From
    // there, NRCTrainingSuffixFinish() walks the suffix backward, turning
    // that single prediction (or zero, for a natural end) into one training
    // record per suffix vertex. To avoid ever dividing by a possibly-tiny
    // throughput (the bug that caused the earlier over-bright render), the
    // suffix tracks its OWN local throughput (nrcSuffixBeta), reset to 1.0 at
    // the render-query vertex and updated purely by forward multiplication --
    // never by dividing the real path's beta.
    //
    // Known, deliberate scope limitations:
    //  - NEE/shadow rays are NOT tracked for suffix vertices (would require
    //    threading new fields through ShadowRayWorkItem across both the CPU
    //    and OptiX shadow-ray kernels); suffix local radiance comes only from
    //    BSDF-sampled continuation and emission hits.
    //  - If the render-query vertex itself (suffix slot 0) lies on an
    //    emitter, that emission is not captured in the suffix's target for
    //    slot 0 (HandleEmissiveIntersection runs before the same vertex's
    //    surfscatter.cpp call that activates suffix tracking).
    //  - Training paths whose query vertex is itself a natural dead end
    //    (invalid BSDF sample / RR-kill) get no suffix and no training
    //    record this pass (a single-vertex record there is always zero, so
    //    dropping it loses nothing).
    static constexpr uint32_t kNRCMaxSuffixLen = 4;
    uint8_t *nrcSuffixActive = nullptr;   // 1 = training path is currently in its suffix; cleared once bootstrap-terminated
    uint8_t *nrcSuffixLen = nullptr;      // number of finalized suffix vertices (valid slots 0..nrcSuffixLen-1)
    uint8_t *nrcSuffixTerminatedByHeuristic = nullptr;  // 1 = suffix ended via its own heuristic/cap (needs a bootstrap query); 0 = natural end
    float *nrcSuffixBeta = nullptr;        // NSpectrumSamples floats/slot: local suffix throughput, reset to 1 at the render-query vertex
    float *nrcSuffixSpreadAccum = nullptr; // suffix's own independent Eq. 3 accumulator
    float *nrcSuffixA0 = nullptr;          // suffix's own independent Eq. 4 baseline, from the real vertex before the render-query vertex to it
    Point3f *nrcSuffixPrevP = nullptr;     // suffix's own "previous vertex" position, seeded from nrcPathPrevP at the render-query vertex, then evolved independently
    float *nrcSuffixPrevPdf = nullptr;     // suffix's own "previous vertex" pdf, seeded from nrcPathPrevPdf at the render-query vertex, then evolved independently
    float *nrcSuffixInputs = nullptr;      // kNRCMaxSuffixLen*kNRCInputDims floats/slot: per-suffix-vertex input feature rows (incl. the bootstrap-only row)
    float *nrcSuffixLocal = nullptr;       // kNRCMaxSuffixLen*NSpectrumSamples floats/slot: per-suffix-vertex local (beta=1-frame) emission contribution
    float *nrcSuffixStep = nullptr;        // kNRCMaxSuffixLen*NSpectrumSamples floats/slot: per-suffix-vertex step factor (f*cos/pdf) to the next vertex
    float *nrcSuffixTarget = nullptr;      // kNRCMaxSuffixLen*kNRCOutputDims floats/slot: backward-propagated RGB target, filled by NRCTrainingSuffixFinish()
    float *nrcSuffixBootstrapInputs = nullptr;  // nrcBatchSize*kNRCInputDims scratch: bootstrap rows gathered contiguously for one Inference() call
    uint32_t nrcCompactCapacity = 0;       // capacity of nrcCompactInputs/nrcCompactTargets in rows (> nrcBatchSize to allow room for suffix records)

    void NRCResetSampleBuffers();
    void NRCCaptureFinalRadiance();
    void NRCTrainAndInferStep();
    void NRCInferenceForRenderPaths();
    void NRCTrainingSuffixFinish();
    void NRCDumpPredictedImage(const std::string &filename);
#endif  // PBRT_BUILD_NRC
};

}  // namespace pbrt

#endif  // PBRT_WAVEFRONT_INTEGRATOR_H
