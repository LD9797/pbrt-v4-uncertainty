// Neural Radiance Cache — NRCPathIntegrator implementation.
// SPDX: Apache-2.0

#include <nrc/nrc_integrator.h>

#include <pbrt/bsdf.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/materials.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/display.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace pbrt {

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// Default record-buffer capacity (~1 M records ≈ 80 MB).
static constexpr int kNRCBufferCapacity = 1 << 20;

// Target clamp to suppress fireflies in training data.
static constexpr float kTargetMaxValue = 100.f;

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
NRCPathIntegrator::NRCPathIntegrator(
    int maxDepth, Camera camera, Sampler sampler,
    Primitive aggregate, std::vector<Light> lights,
    const std::string &lightSampleStrategy, bool regularize,
    int recordDepth, bool usePrediction, bool writeDebugImage,
    int nHiddenLayers, int nNeurons)
    : RayIntegrator(camera, sampler, aggregate, lights),
      maxDepth_(maxDepth),
      recordDepth_(recordDepth),
      regularize_(regularize),
      usePrediction_(usePrediction),
      writeDebugImage_(writeDebugImage),
      lightSampler_(
          LightSampler::Create(lightSampleStrategy, lights, Allocator())),
      sceneBounds_(aggregate ? aggregate.Bounds() : Bounds3f()),
      nrcBuffer_(std::make_unique<NRCRecordBuffer>(kNRCBufferCapacity)),
      trainer_(std::make_unique<NRCTrainer>(nHiddenLayers, nNeurons)) {

    LOG_VERBOSE("NRCPathIntegrator: recordDepth=%d usePrediction=%s",
                recordDepth_, usePrediction_ ? "true" : "false");
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────
std::unique_ptr<NRCPathIntegrator> NRCPathIntegrator::Create(
    const ParameterDictionary &parameters, Camera camera, Sampler sampler,
    Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {

    int  maxDepth        = parameters.GetOneInt("maxdepth",        5);
    int  recordDepth     = parameters.GetOneInt("recorddepth",     2);
    bool usePrediction   = parameters.GetOneBool("useprediction",  false);
    bool writeDebugImage = parameters.GetOneBool("debugimage",     false);
    int  nHiddenLayers   = parameters.GetOneInt("nhiddenlayers",   4);
    int  nNeurons        = parameters.GetOneInt("nneurons",        64);
    std::string ls       = parameters.GetOneString("lightsampler", "bvh");
    bool regularize      = parameters.GetOneBool("regularize",     false);

    return std::make_unique<NRCPathIntegrator>(
        maxDepth, camera, sampler, aggregate, lights,
        ls, regularize, recordDepth, usePrediction, writeDebugImage,
        nHiddenLayers, nNeurons);
}

std::string NRCPathIntegrator::ToString() const {
    return StringPrintf(
        "[ NRCPathIntegrator maxDepth: %d recordDepth: %d "
        "usePrediction: %s lightSampler: %s regularize: %s ]",
        maxDepth_, recordDepth_,
        usePrediction_ ? "true" : "false",
        lightSampler_, regularize_ ? "true" : "false");
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3 helper — encode surface vertex into 16-D NRC input features
// ─────────────────────────────────────────────────────────────────────────────
void NRCPathIntegrator::FillInputFeatures(
    float *f16,
    const SurfaceInteraction &isect,
    const BSDF &bsdf,
    const SampledWavelengths &lambda) const {

    // ── Position: normalised to [0,1] via scene AABB ─────────────────────────
    Point3f p = isect.p();
    auto norm = [&](int axis, float v) -> float {
        float lo = sceneBounds_.pMin[axis];
        float hi = sceneBounds_.pMax[axis];
        float range = hi - lo;
        return (range > 0.f) ? (v - lo) / range : 0.5f;
    };
    f16[0] = norm(0, p.x);
    f16[1] = norm(1, p.y);
    f16[2] = norm(2, p.z);

    // ── Shading normal ────────────────────────────────────────────────────────
    Normal3f n = isect.shading.n;
    f16[3] = n.x;  f16[4] = n.y;  f16[5] = n.z;

    // ── Outgoing direction wo ─────────────────────────────────────────────────
    Vector3f wo = isect.wo;
    f16[6] = wo.x;  f16[7] = wo.y;  f16[8] = wo.z;

    // ── Diffuse/hemispherical-directional reflectance ─────────────────────────
    // Use the same fixed sample arrays as PathIntegrator's visible-surface init.
    static const float ucRho[16] = {
        0.75741637f, 0.37870818f, 0.7083487f,  0.18935409f,
        0.9149363f,  0.35417435f, 0.5990858f,  0.09467703f,
        0.8578725f,  0.45746812f, 0.686759f,   0.17708716f,
        0.9674518f,  0.2995429f,  0.5083201f,  0.047338516f};
    static const Point2f uRho[16] = {
        {0.855985f,0.570367f},{0.381823f,0.851844f},{0.285328f,0.764262f},
        {0.733380f,0.114073f},{0.542663f,0.344465f},{0.127274f,0.414848f},
        {0.964700f,0.947162f},{0.594089f,0.643463f},{0.095109f,0.170369f},
        {0.825444f,0.263359f},{0.429467f,0.454469f},{0.244460f,0.816459f},
        {0.756135f,0.731258f},{0.516165f,0.152852f},{0.180888f,0.214174f},
        {0.898579f,0.503897f}};

    SampledSpectrum rho = bsdf.rho(isect.wo, ucRho, uRho);
    RGB albedo = rho.ToRGB(lambda, *RGBColorSpace::sRGB);
    f16[9]  = Clamp(albedo.r, 0.f, 1.f);
    f16[10] = Clamp(albedo.g, 0.f, 1.f);
    f16[11] = Clamp(albedo.b, 0.f, 1.f);

    // ── Roughness proxy (placeholder) + padding ───────────────────────────────
    f16[12] = 0.5f;   // TODO: query material roughness
    f16[13] = 0.f;    // reserved: specular R
    f16[14] = 0.f;    // reserved: specular G
    f16[15] = 0.f;    // reserved: specular B
}

// ─────────────────────────────────────────────────────────────────────────────
// Direct-lighting helper — mirrors PathIntegrator::SampleLd
// ─────────────────────────────────────────────────────────────────────────────
SampledSpectrum NRCPathIntegrator::SampleLd(
    const SurfaceInteraction &intr, const BSDF *bsdf,
    SampledWavelengths &lambda, Sampler sampler) const {

    LightSampleContext ctx(intr);
    BxDFFlags flags = bsdf->Flags();
    if (IsReflective(flags) && !IsTransmissive(flags))
        ctx.pi = intr.OffsetRayOrigin(intr.wo);
    else if (IsTransmissive(flags) && !IsReflective(flags))
        ctx.pi = intr.OffsetRayOrigin(-intr.wo);

    Float u = sampler.Get1D();
    pstd::optional<SampledLight> sampledLight = lightSampler_.Sample(ctx, u);
    Point2f uLight = sampler.Get2D();
    if (!sampledLight) return {};

    Light light = sampledLight->light;
    pstd::optional<LightLiSample> ls =
        light.SampleLi(ctx, uLight, lambda, true);
    if (!ls || !ls->L || ls->pdf == 0) return {};

    Vector3f wo = intr.wo, wi = ls->wi;
    SampledSpectrum f = bsdf->f(wo, wi) * AbsDot(wi, intr.shading.n);
    if (!f || !Unoccluded(intr, ls->pLight)) return {};

    Float p_l = sampledLight->p * ls->pdf;
    if (IsDeltaLight(light.Type()))
        return ls->L * f / p_l;

    Float p_b = bsdf->PDF(wo, wi);
    Float w_l = PowerHeuristic(1, p_l, 1, p_b);
    return w_l * ls->L * f / p_l;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3 — Li(): path tracing with NRC record collection
// Steps 5,7 are also implemented here
// ─────────────────────────────────────────────────────────────────────────────
SampledSpectrum NRCPathIntegrator::Li(
    RayDifferential ray, SampledWavelengths &lambda,
    Sampler sampler, ScratchBuffer &scratchBuffer,
    VisibleSurface *visibleSurf) const {

    // ── Path state ────────────────────────────────────────────────────────────
    SampledSpectrum L(0.f), beta(1.f);
    int depth = 0;
    Float p_b = 0.f, etaScale = 1.f;
    bool specularBounce = false, anyNonSpecularBounces = false;
    LightSampleContext prevIntrCtx;

    // ── NRC record state ──────────────────────────────────────────────────────
    int         nrcIdx     = -1;          // record slot (-1 = not yet allocated)
    SampledSpectrum nrcLBefore(0.f);     // L accumulated before the cache point
    SampledSpectrum nrcBeta(1.f);        // beta at the cache point

    // ── Path loop ─────────────────────────────────────────────────────────────
    while (true) {
        pstd::optional<ShapeIntersection> si = Intersect(ray);

        // Escaped ray — add infinite-light emission
        if (!si) {
            for (const auto &light : infiniteLights) {
                SampledSpectrum Le = light.Le(ray, lambda);
                if (depth == 0 || specularBounce)
                    L += beta * Le;
                else {
                    Float p_l = lightSampler_.PMF(prevIntrCtx, light) *
                                light.PDF_Li(prevIntrCtx, ray.d, true);
                    L += beta * PowerHeuristic(1, p_b, 1, p_l) * Le;
                }
            }
            break;
        }

        // Surface emission
        SampledSpectrum Le = si->intr.Le(-ray.d, lambda);
        if (Le) {
            if (depth == 0 || specularBounce)
                L += beta * Le;
            else {
                Light areaLight(si->intr.areaLight);
                Float p_l = lightSampler_.PMF(prevIntrCtx, areaLight) *
                            areaLight.PDF_Li(prevIntrCtx, ray.d, true);
                L += beta * PowerHeuristic(1, p_b, 1, p_l) * Le;
            }
        }

        SurfaceInteraction &isect = si->intr;

        // BSDF / medium boundary
        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
        if (!bsdf) {
            specularBounce = true;
            isect.SkipIntersection(&ray, si->tHit);
            continue;
        }

        // Initialise visible surface on first hit
        if (depth == 0 && visibleSurf) {
            constexpr int nRhoSamples = 16;
            const Float ucRho[nRhoSamples] = {
                0.75741637f,0.37870818f,0.7083487f, 0.18935409f,
                0.9149363f, 0.35417435f,0.5990858f, 0.09467703f,
                0.8578725f, 0.45746812f,0.686759f,  0.17708716f,
                0.9674518f, 0.2995429f, 0.5083201f, 0.047338516f};
            const Point2f uRho[nRhoSamples] = {
                {0.855985f,0.570367f},{0.381823f,0.851844f},
                {0.285328f,0.764262f},{0.733380f,0.114073f},
                {0.542663f,0.344465f},{0.127274f,0.414848f},
                {0.964700f,0.947162f},{0.594089f,0.643463f},
                {0.095109f,0.170369f},{0.825444f,0.263359f},
                {0.429467f,0.454469f},{0.244460f,0.816459f},
                {0.756135f,0.731258f},{0.516165f,0.152852f},
                {0.180888f,0.214174f},{0.898579f,0.503897f}};
            *visibleSurf = VisibleSurface(
                isect, bsdf.rho(isect.wo, ucRho, uRho), lambda);
        }

        if (regularize_ && anyNonSpecularBounces) bsdf.Regularize();

        // ── Step 3: record NRC input at recordDepth_ ──────────────────────────
        if (depth == recordDepth_ && nrcIdx == -1) {
            nrcIdx = nrcBuffer_->Alloc();
            if (nrcIdx >= 0) {
                FillInputFeatures((*nrcBuffer_)[nrcIdx].input, isect, bsdf, lambda);
                nrcLBefore = L;
                nrcBeta    = beta;
            }
        }

        // ── Step 7: if NRC is warmed up, terminate path and query NRC ─────────
        if (usePrediction_ && depth == recordDepth_ &&
            trainer_->TrainSteps() > 0) {

            if (nrcIdx >= 0) {
                // Run inference for just this one vertex.
                float *inp = (*nrcBuffer_)[nrcIdx].input;
                float  out[NRC_OUTPUT_DIMS] = {0.f, 0.f, 0.f};
                trainer_->Inference(inp, 1, out);

                // Reconstruct the NRC prediction as a SampledSpectrum via RGB.
                // For path integration: L += beta * L_nrc.
                // We approximate L_nrc as a flat spectrum matching the RGB values.
                // RGBUnboundedSpectrum handles values > 1 (radiance, not albedo)
                SampledSpectrum nrcPred =
                    RGBUnboundedSpectrum(
                        *RGBColorSpace::sRGB,
                        RGB(std::max(out[0], 0.f),
                            std::max(out[1], 0.f),
                            std::max(out[2], 0.f)))
                    .Sample(lambda);

                L += beta * nrcPred;
                // Mark record so training skips this (no valid target).
                (*nrcBuffer_)[nrcIdx].targetValid = false;
            }
            break;  // path terminated — NRC takes over
        }

        // ── Continue standard path tracing ────────────────────────────────────
        if (depth++ == maxDepth_) break;

        if (IsNonSpecular(bsdf.Flags()))
            L += beta * SampleLd(isect, &bsdf, lambda, sampler);

        // BSDF sampling for next direction
        Vector3f wo = -ray.d;
        Float u = sampler.Get1D();
        pstd::optional<BSDFSample> bs = bsdf.Sample_f(wo, u, sampler.Get2D());
        if (!bs) break;

        beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
        p_b  = bs->pdfIsProportional ? bsdf.PDF(wo, bs->wi) : bs->pdf;
        DCHECK(!IsInf(beta.y(lambda)));
        specularBounce         = bs->IsSpecular();
        anyNonSpecularBounces |= !bs->IsSpecular();
        if (bs->IsTransmission()) etaScale *= Sqr(bs->eta);
        prevIntrCtx = si->intr;

        ray = isect.SpawnRay(ray, bsdf, bs->wi, bs->flags, bs->eta);

        // Russian roulette
        SampledSpectrum rrBeta = beta * etaScale;
        if (rrBeta.MaxComponentValue() < 1.f && depth > 1) {
            Float q = std::max<Float>(0.f, 1.f - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q) break;
            beta /= 1.f - q;
            DCHECK(!IsInf(beta.y(lambda)));
        }
    }

    // ── Step 3: fill NRC target ───────────────────────────────────────────────
    // target = (L_total - L_before_cache) / betaLum
    // This gives the incident radiance estimate at the cache point.
    if (nrcIdx >= 0 && !usePrediction_) {
        NRCRecord &rec = (*nrcBuffer_)[nrcIdx];
        Float betaLum  = nrcBeta.y(lambda);
        if (betaLum > 1e-4f) {
            SampledSpectrum lRest = (L - nrcLBefore) * (1.f / betaLum);
            RGB target = lRest.ToRGB(lambda, *RGBColorSpace::sRGB);
            rec.target[0] = Clamp(target.r, 0.f, kTargetMaxValue);
            rec.target[1] = Clamp(target.g, 0.f, kTargetMaxValue);
            rec.target[2] = Clamp(target.b, 0.f, kTargetMaxValue);
            rec.targetValid = true;
        }
    }

    return L;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5 — print a few target/prediction examples
// ─────────────────────────────────────────────────────────────────────────────
void NRCPathIntegrator::PrintExamples(float loss) const {
    int count = nrcBuffer_->Count();
    int shown = 0;
    Printf("[NRC] train step %d — %d records — loss = %.6f\n",
           trainer_->TrainSteps(), count, loss);

    if (count == 0) return;

    // Gather up to 5 valid records for inference.
    static constexpr int kExamples = 5;
    std::vector<int> exampleIndices;
    for (int i = 0; i < count && (int)exampleIndices.size() < kExamples; ++i)
        if ((*nrcBuffer_)[i].targetValid) exampleIndices.push_back(i);

    if (exampleIndices.empty()) return;

    int n = (int)exampleIndices.size();
    std::vector<float> inputs(n * NRC_INPUT_DIMS);
    for (int k = 0; k < n; ++k) {
        const float *src = (*nrcBuffer_)[exampleIndices[k]].input;
        std::copy(src, src + NRC_INPUT_DIMS, inputs.data() + k * NRC_INPUT_DIMS);
    }

    std::vector<float> preds(n * NRC_OUTPUT_DIMS, 0.f);
    trainer_->Inference(inputs.data(), n, preds.data());

    Printf("[NRC] target → prediction examples:\n");
    for (int k = 0; k < n; ++k) {
        const float *t = (*nrcBuffer_)[exampleIndices[k]].target;
        const float *p = preds.data() + k * NRC_OUTPUT_DIMS;
        Printf("  [%.3f %.3f %.3f] → [%.3f %.3f %.3f]\n",
               t[0], t[1], t[2], p[0], p[1], p[2]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 6 — write debug_nrc.exr (NRC-only radiance prediction)
// ─────────────────────────────────────────────────────────────────────────────
void NRCPathIntegrator::WriteDebugImage() {
    if (trainer_->TrainSteps() == 0) return;

    Film film         = camera.GetFilm();
    Bounds2i bounds   = film.PixelBounds();
    Point2i  res      = film.FullResolution();
    int      width    = bounds.pMax.x - bounds.pMin.x;
    int      height   = bounds.pMax.y - bounds.pMin.y;

    // One thread per pixel: trace one camera ray → first hit → NRC query.
    std::vector<float> imageData(width * height * 3, 0.f);

    ThreadLocal<ScratchBuffer> scratchBuffers([] { return ScratchBuffer(); });
    Sampler samplerProto = samplerPrototype;  // non-const copy for Clone()
    ThreadLocal<Sampler> samplers([&samplerProto] { return samplerProto.Clone(); });

    Filter filter = film.GetFilter();

    ParallelFor2D(bounds, [&](Bounds2i tile) {
        ScratchBuffer &scratchBuf  = scratchBuffers.Get();
        Sampler       &tileSampler = samplers.Get();

        for (Point2i pPixel : tile) {
            tileSampler.StartPixelSample(pPixel, 0);
            Float lu = tileSampler.Get1D();
            SampledWavelengths lambda = film.SampleWavelengths(lu);
            CameraSample cs = GetCameraSample(tileSampler, pPixel, filter);
            pstd::optional<CameraRayDifferential> cr =
                camera.GenerateRayDifferential(cs, lambda);
            if (!cr) continue;

            RayDifferential ray = cr->ray;
            pstd::optional<ShapeIntersection> si = Intersect(ray);
            if (!si) continue;

            SurfaceInteraction &isect = si->intr;
            BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuf, tileSampler);
            if (!bsdf) continue;

            float inp[NRC_INPUT_DIMS];
            FillInputFeatures(inp, isect, bsdf, lambda);

            float out[NRC_OUTPUT_DIMS] = {0.f, 0.f, 0.f};
            trainer_->Inference(inp, 1, out);

            int px = pPixel.x - bounds.pMin.x;
            int py = pPixel.y - bounds.pMin.y;
            int base = (py * width + px) * 3;
            imageData[base + 0] = std::max(out[0], 0.f);
            imageData[base + 1] = std::max(out[1], 0.f);
            imageData[base + 2] = std::max(out[2], 0.f);

            scratchBuf.Reset();
        }
    });

    // Write via PBRT's Image class.
    std::vector<std::string> channelNames = {"R", "G", "B"};
    Image img(PixelFormat::Float, {width, height}, channelNames);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c)
                img.SetChannel({x, y}, c, imageData[(y * width + x) * 3 + c]);

    ImageMetadata meta;
    meta.pixelBounds = bounds;
    meta.fullResolution = res;

    std::string filename = "debug_nrc.exr";
    img.Write(filename, meta);
    Printf("[NRC] debug image written to %s\n", filename);
}

// ─────────────────────────────────────────────────────────────────────────────
// Steps 4–6 — Render() override: trace → train → print → debug image
// ─────────────────────────────────────────────────────────────────────────────
void NRCPathIntegrator::Render() {
    // ── Reset NRC buffer from any previous call ────────────────────────────────
    nrcBuffer_->Reset();

    // ── Step 3: trace all paths, collecting NRC records via Li() ──────────────
    ImageTileIntegrator::Render();

    // ── Step 4: train tiny-cuda-nn ────────────────────────────────────────────
    float loss = trainer_->Train(*nrcBuffer_);
    if (loss < 0.f) {
        Warning("NRC: not enough valid records to train (%d collected). "
                "Increase spp or reduce recorddepth.",
                nrcBuffer_->Count());
        return;
    }

    // ── Step 5: print loss and examples ───────────────────────────────────────
    PrintExamples(loss);

    // ── Step 6: debug image ───────────────────────────────────────────────────
    if (writeDebugImage_)
        WriteDebugImage();
}

}  // namespace pbrt
