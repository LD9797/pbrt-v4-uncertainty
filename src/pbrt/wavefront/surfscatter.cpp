// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/bxdfs.h>
#include <pbrt/cameras.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/options.h>
#include <pbrt/textures.h>
#include <pbrt/util/check.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/media.h>
#include <pbrt/wavefront/integrator.h>

#include <type_traits>

namespace pbrt {

// EvaluateMaterialCallback Definition
struct EvaluateMaterialCallback {
    int wavefrontDepth;
    WavefrontPathIntegrator *integrator;
    Transform movingFromCamera;
    // EvaluateMaterialCallback Public Methods
    template <typename ConcreteMaterial>
    void operator()() {
        if constexpr (!std::is_same_v<ConcreteMaterial, MixMaterial>)
            integrator->EvaluateMaterialAndBSDF<ConcreteMaterial>(wavefrontDepth,
                                                                  movingFromCamera);
    }
};

// WavefrontPathIntegrator Surface Scattering Methods
void WavefrontPathIntegrator::EvaluateMaterialsAndBSDFs(int wavefrontDepth,
                                                        Transform movingFromCamera) {
    ForEachType(EvaluateMaterialCallback{wavefrontDepth, this, movingFromCamera},
                Material::Types());
}

template <typename ConcreteMaterial>
void WavefrontPathIntegrator::EvaluateMaterialAndBSDF(int wavefrontDepth,
                                                      Transform movingFromCamera) {
    int index = Material::TypeIndex<ConcreteMaterial>();
    if (haveBasicEvalMaterial[index])
        EvaluateMaterialAndBSDF<ConcreteMaterial, BasicTextureEvaluator>(
            basicEvalMaterialQueue, movingFromCamera, wavefrontDepth);
    if (haveUniversalEvalMaterial[index])
        EvaluateMaterialAndBSDF<ConcreteMaterial, UniversalTextureEvaluator>(
            universalEvalMaterialQueue, movingFromCamera, wavefrontDepth);
}

template <typename ConcreteMaterial, typename TextureEvaluator>
void WavefrontPathIntegrator::EvaluateMaterialAndBSDF(MaterialEvalQueue *evalQueue,
                                                      Transform movingFromCamera,
                                                      int wavefrontDepth) {
    // Get BSDF for items in _evalQueue_ and sample illumination
    // Construct _desc_ for material/texture evaluation kernel
    std::string desc = StringPrintf(
        "%s + BxDF eval (%s tex)", ConcreteMaterial::Name(),
        std::is_same_v<TextureEvaluator, BasicTextureEvaluator> ? "Basic" : "Universal");

    RayQueue *nextRayQueue = NextRayQueue(wavefrontDepth);
    auto queue = evalQueue->Get<MaterialEvalWorkItem<ConcreteMaterial>>();
    ForAllQueued(
        desc.c_str(), queue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const MaterialEvalWorkItem<ConcreteMaterial> w) {
            // Evaluate material and BSDF for ray intersection
            TextureEvaluator texEval;
            // Compute differentials for position and $(u,v)$ at intersection point
            Vector3f dpdx, dpdy;
            Float dudx = 0, dudy = 0, dvdx = 0, dvdy = 0;
            if (!GetOptions().disableTextureFiltering) {
                Point3f pc = movingFromCamera.ApplyInverse(Point3f(w.pi));
                Normal3f nc = movingFromCamera.ApplyInverse(w.n);
                camera.Approximate_dp_dxy(pc, nc, w.time, samplesPerPixel, &dpdx, &dpdy);
                Vector3f dpdu = w.dpdu, dpdv = w.dpdv;
                // Estimate screen-space change in $(u,v)$
                // Compute $\transpose{\XFORM{A}} \XFORM{A}$ and its determinant
                Float ata00 = Dot(dpdu, dpdu), ata01 = Dot(dpdu, dpdv);
                Float ata11 = Dot(dpdv, dpdv);
                Float invDet = 1 / DifferenceOfProducts(ata00, ata11, ata01, ata01);
                invDet = IsFinite(invDet) ? invDet : 0.f;

                // Compute $\transpose{\XFORM{A}} \VEC{b}$ for $x$ and $y$
                Float atb0x = Dot(dpdu, dpdx), atb1x = Dot(dpdv, dpdx);
                Float atb0y = Dot(dpdu, dpdy), atb1y = Dot(dpdv, dpdy);

                // Compute $u$ and $v$ derivatives with respect to $x$ and $y$
                dudx = DifferenceOfProducts(ata11, atb0x, ata01, atb1x) * invDet;
                dvdx = DifferenceOfProducts(ata00, atb1x, ata01, atb0x) * invDet;
                dudy = DifferenceOfProducts(ata11, atb0y, ata01, atb1y) * invDet;
                dvdy = DifferenceOfProducts(ata00, atb1y, ata01, atb0y) * invDet;

                // Clamp derivatives of $u$ and $v$ to reasonable values
                dudx = IsFinite(dudx) ? Clamp(dudx, -1e8f, 1e8f) : 0.f;
                dvdx = IsFinite(dvdx) ? Clamp(dvdx, -1e8f, 1e8f) : 0.f;
                dudy = IsFinite(dudy) ? Clamp(dudy, -1e8f, 1e8f) : 0.f;
                dvdy = IsFinite(dvdy) ? Clamp(dvdy, -1e8f, 1e8f) : 0.f;
            }

            // Compute shading normal if bump or normal mapping is being used
            Normal3f ns = w.ns;
            Vector3f dpdus = w.dpdus;
            FloatTexture displacement = w.material->GetDisplacement();
            const Image *normalMap = w.material->GetNormalMap();
            if (normalMap) {
                // Call _NormalMap()_ to find shading geometry
                NormalBumpEvalContext bctx =
                    w.GetNormalBumpEvalContext(dudx, dudy, dvdx, dvdy);
                Vector3f dpdvs;
                NormalMap(*normalMap, bctx, &dpdus, &dpdvs);
                ns = Normal3f(Normalize(Cross(dpdus, dpdvs)));
                ns = FaceForward(ns, w.n);

            } else if (displacement) {
                // Call _BumpMap()_ to find shading geometry
                if (displacement)
                    DCHECK(texEval.CanEvaluate({displacement}, {}));
                NormalBumpEvalContext bctx =
                    w.GetNormalBumpEvalContext(dudx, dudy, dvdx, dvdy);
                Vector3f dpdvs;
                BumpMap(texEval, displacement, bctx, &dpdus, &dpdvs);
                ns = Normal3f(Normalize(Cross(dpdus, dpdvs)));
                ns = FaceForward(ns, w.n);
            }

            // Get BSDF at intersection point
            SampledWavelengths lambda = w.lambda;
            MaterialEvalContext ctx =
                w.GetMaterialEvalContext(dudx, dudy, dvdx, dvdy, ns, dpdus);
            using ConcreteBxDF = typename ConcreteMaterial::BxDF;
            ConcreteBxDF bxdf = w.material->GetBxDF(texEval, ctx, lambda);
            BSDF bsdf(ctx.ns, ctx.dpdus, &bxdf);
            // Handle terminated secondary wavelengths after BSDF creation
            if (lambda.SecondaryTerminated())
                pixelSampleState.lambda[w.pixelIndex] = lambda;

            // Regularize BSDF, if appropriate
            if (regularize && w.anyNonSpecularBounces)
                bsdf.Regularize();

#ifdef PBRT_BUILD_NRC
            // NRC milestone 3: track the Muller et al. 2021 area-spread path
            // termination heuristic (Sec. 3.4) to choose the query vertex
            // dynamically per path -- see the long comment on
            // kNRCSpreadC/nrcPathSpreadAccum in integrator.h for the exact
            // formulas. Per Muller et al., "all paths are terminated according
            // to" this heuristic -- it governs every rendering path, not just
            // the sparse subset selected to generate NRC training records.
            // nrcTrainingPath only gates whether the *query vertex we find*
            // gets written into nrcInputs/nrcTargets below.
            bool nrcSpreadActive = nrcInputs != nullptr && !nrcValid[w.pixelIndex];
            // Set to true once this vertex is determined to be the query
            // vertex, either because the accumulated spread crossed the
            // threshold or because the path is about to end for some other
            // reason (max depth, Russian roulette, no valid BSDF sample).
            bool nrcCaptureNow = false;
            if (nrcSpreadActive) {
                Point3f p(w.pi);
                Float cosThetaHere = AbsDot(w.wo, ns);
                if (w.depth == 0) {
                    // Primary vertex x1: establish the Eq. 4 baseline a0 from
                    // the camera-to-x1 distance, and start the Eq. 3 spread
                    // sum fresh (it accumulates segments x1-x2, x2-x3, ...).
                    Float d1Sq = DistanceSquared(nrcPathPrevP[w.pixelIndex], p);
                    nrcPathA0[w.pixelIndex] =
                        d1Sq / (4 * Pi * std::max<Float>(cosThetaHere, 1e-6f));
                    nrcPathSpreadAccum[w.pixelIndex] = 0.f;
                } else {
                    // Vertex x_i, i = depth+1 >= 2: accumulate this segment's
                    // contribution to Eq. 3 and test against c * a0.
                    Float dSq = DistanceSquared(nrcPathPrevP[w.pixelIndex], p);
                    Float pdfPrev = std::max<Float>(nrcPathPrevPdf[w.pixelIndex], 1e-6f);
                    Float term =
                        std::sqrt(dSq / (pdfPrev * std::max<Float>(cosThetaHere, 1e-6f)));
                    Float accum = nrcPathSpreadAccum[w.pixelIndex] + term;
                    nrcPathSpreadAccum[w.pixelIndex] = accum;
                    if (Sqr(accum) > kNRCSpreadC * nrcPathA0[w.pixelIndex])
                        nrcCaptureNow = true;
                }
                // Not the only way a path can stop: this is also the last
                // vertex EvaluateMaterialsAndBSDFs will ever process for this
                // ray, since the wavefront loop breaks once wavefrontDepth ==
                // maxDepth (see Render()) without shading that final hit.
                if (w.depth == maxDepth - 1)
                    nrcCaptureNow = true;
            }
#endif

            // Initialize _VisibleSurface_ at first intersection if necessary
            if (w.depth == 0 && initializeVisibleSurface) {
                SurfaceInteraction isect;
                isect.pi = w.pi;
                isect.n = w.n;
                isect.shading.n = ns;
                isect.uv = w.uv;
                isect.wo = w.wo;
                isect.time = w.time;
                isect.dpdx = dpdx;
                isect.dpdy = dpdy;

                // Estimate BSDF's albedo
                // Define sample arrays _ucRho_ and _uRho_ for reflectance estimate
                constexpr int nRhoSamples = 16;
                const Float ucRho[nRhoSamples] = {
                    0.75741637, 0.37870818, 0.7083487, 0.18935409, 0.9149363, 0.35417435,
                    0.5990858,  0.09467703, 0.8578725, 0.45746812, 0.686759,  0.17708716,
                    0.9674518,  0.2995429,  0.5083201, 0.047338516};
                const Point2f uRho[nRhoSamples] = {
                    Point2f(0.855985, 0.570367), Point2f(0.381823, 0.851844),
                    Point2f(0.285328, 0.764262), Point2f(0.733380, 0.114073),
                    Point2f(0.542663, 0.344465), Point2f(0.127274, 0.414848),
                    Point2f(0.964700, 0.947162), Point2f(0.594089, 0.643463),
                    Point2f(0.095109, 0.170369), Point2f(0.825444, 0.263359),
                    Point2f(0.429467, 0.454469), Point2f(0.244460, 0.816459),
                    Point2f(0.756135, 0.731258), Point2f(0.516165, 0.152852),
                    Point2f(0.180888, 0.214174), Point2f(0.898579, 0.503897)};

                SampledSpectrum albedo = bsdf.rho(isect.wo, ucRho, uRho);

                pixelSampleState.visibleSurface[w.pixelIndex] =
                    VisibleSurface(isect, albedo, lambda);
            }

            // Sample BSDF and enqueue indirect ray at intersection point
            Vector3f wo = w.wo;
            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            pstd::optional<BSDFSample> bsdfSample = bsdf.Sample_f<ConcreteBxDF>(
                wo, raySamples.indirect.uc, raySamples.indirect.u);
#ifdef PBRT_BUILD_NRC
            // No valid outgoing direction: the path ends at this vertex, so
            // it becomes the query vertex if nothing has captured yet.
            if (nrcSpreadActive && !bsdfSample)
                nrcCaptureNow = true;
#endif
            if (bsdfSample) {
                // Compute updated path throughput and PDFs and enqueue indirect ray
                Vector3f wi = bsdfSample->wi;
                SampledSpectrum beta =
                    w.beta * bsdfSample->f * AbsDot(wi, ns) / bsdfSample->pdf;
                SampledSpectrum r_u = w.r_u, r_l;

                PBRT_DBG("%s f*cos[0] %f bsdfSample->pdf %f f*cos/pdf %f\n",
                         ConcreteBxDF::Name(), bsdfSample->f[0] * AbsDot(wi, ns),
                         bsdfSample->pdf,
                         bsdfSample->f[0] * AbsDot(wi, ns) / bsdfSample->pdf);

                // Update _r_u_ based on BSDF sample PDF
                if (bsdfSample->pdfIsProportional)
                    r_l = r_u / bsdf.PDF<ConcreteBxDF>(wo, bsdfSample->wi);
                else
                    r_l = r_u / bsdfSample->pdf;

                // Update _etaScale_ accounting for BSDF scattering
                Float etaScale = w.etaScale;
                if (bsdfSample->IsTransmission())
                    etaScale *= Sqr(bsdfSample->eta);

                // Apply Russian roulette to indirect ray based on weighted path
                // throughput
                SampledSpectrum rrBeta = beta * etaScale / r_u.Average();
                // Note: depth >= 1 here to match VolPathIntegrator (which increments
                // depth earlier).
                if (rrBeta.MaxComponentValue() < 1 && w.depth >= 1) {
                    Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                    if (raySamples.indirect.rr < q) {
                        beta = SampledSpectrum(0.f);
                        PBRT_DBG("Path terminated with RR\n");
                    } else
                        beta /= 1 - q;
                }

#ifdef PBRT_BUILD_NRC
                // Russian roulette (or f=0/pdf=0 upstream) killed the path:
                // no further vertex will exist, so this one becomes the query
                // vertex if nothing has captured yet.
                if (nrcSpreadActive && !beta)
                    nrcCaptureNow = true;
#endif

                if (beta) {
                    // Initialize spawned ray and enqueue for next ray depth
                    if (bsdfSample->IsTransmission() &&
                        w.material->HasSubsurfaceScattering()) {
                        bssrdfEvalQueue->Push(w.material, lambda, beta, r_u,
                                              Point3f(w.pi), wo, w.n, ns, dpdus, w.uv,
                                              w.depth, w.mediumInterface, etaScale,
                                              w.pixelIndex);
                    } else {
                        Ray ray = SpawnRay(w.pi, w.n, w.time, wi);
                        // Initialize _ray_ medium if media are present
                        if (haveMedia)
                            ray.medium = Dot(ray.d, w.n) > 0 ? w.mediumInterface.outside
                                                             : w.mediumInterface.inside;

                        bool anyNonSpecularBounces =
                            !bsdfSample->IsSpecular() || w.anyNonSpecularBounces;
                        // NOTE: slightly different than context below. Problem?
                        LightSampleContext ctx(w.pi, w.n, ns);
                        nextRayQueue->PushIndirectRay(
                            ray, w.depth + 1, ctx, beta, r_u, r_l, lambda,
                            etaScale, bsdfSample->IsSpecular(), anyNonSpecularBounces,
                            w.pixelIndex);

                        PBRT_DBG(
                            "Spawned indirect ray at depth %d from w.index %d. "
                            "Specular %d beta %f %f %f %f r_u %f %f %f %f r_l %f "
                            "%f %f %f beta/r_u %f %f %f %f\n",
                            w.depth + 1, w.pixelIndex, int(bsdfSample->IsSpecular()),
                            beta[0], beta[1], beta[2], beta[3], r_u[0], r_u[1],
                            r_u[2], r_u[3], r_l[0], r_l[1], r_l[2],
                            r_l[3], SafeDiv(beta, r_u)[0],
                            SafeDiv(beta, r_u)[1], SafeDiv(beta, r_u)[2],
                            SafeDiv(beta, r_u)[3]);
                    }

#ifdef PBRT_BUILD_NRC
                    // Path continues past this vertex: remember it as x_{i-1}
                    // and the pdf used to sample wi as p(w_i | x_{i-1}) for
                    // the next vertex's Eq. 3 segment contribution.
                    if (nrcSpreadActive && !nrcCaptureNow) {
                        nrcPathPrevP[w.pixelIndex] = Point3f(w.pi);
                        nrcPathPrevPdf[w.pixelIndex] =
                            bsdfSample->pdfIsProportional
                                ? bsdf.PDF<ConcreteBxDF>(wo, bsdfSample->wi)
                                : bsdfSample->pdf;
                    }
#endif
                }
            }

#ifdef PBRT_BUILD_NRC
            // This vertex was chosen as the NRC query vertex (see the
            // area-spread tracking block above): capture its feature vector,
            // matching Muller et al. 2021's 64-wide input layer 1:1. Inputs
            // are stored column-major: kNRCInputDims raw floats per slot
            // (encoding to the full 64 dims happens in nrc_config.json),
            // pixelIndex==slot. Only paths selected as NRC training paths
            // (nrcTrainingPath, set in GenerateCameraRays -- 1 out of every
            // 32, or all of them during the final inference sweep) actually
            // write a record; other paths still ran the heuristic above (to
            // match Muller et al.'s termination rule) but don't capture.
            if (nrcSpreadActive && nrcCaptureNow && nrcTrainingPath[w.pixelIndex]) {
                // Albedo: hemispherical-directional reflectance.
                constexpr int nRhoSamples = 16;
                const Float ucRho[nRhoSamples] = {
                    0.75741637, 0.37870818, 0.7083487, 0.18935409, 0.9149363, 0.35417435,
                    0.5990858,  0.09467703, 0.8578725, 0.45746812, 0.686759,  0.17708716,
                    0.9674518,  0.2995429,  0.5083201, 0.047338516};
                const Point2f uRho[nRhoSamples] = {
                    Point2f(0.855985, 0.570367), Point2f(0.381823, 0.851844),
                    Point2f(0.285328, 0.764262), Point2f(0.733380, 0.114073),
                    Point2f(0.542663, 0.344465), Point2f(0.127274, 0.414848),
                    Point2f(0.964700, 0.947162), Point2f(0.594089, 0.643463),
                    Point2f(0.095109, 0.170369), Point2f(0.825444, 0.263359),
                    Point2f(0.429467, 0.454469), Point2f(0.244460, 0.816459),
                    Point2f(0.756135, 0.731258), Point2f(0.516165, 0.152852),
                    Point2f(0.180888, 0.214174), Point2f(0.898579, 0.503897)};
                SampledSpectrum albedo = bsdf.rho(wo, ucRho, uRho);

                Point3f p(w.pi);
                float *row = nrcInputs + size_t(w.pixelIndex) * kNRCInputDims;
                // dims 0-35: position, normalized to [0,1] via scene bounds, then encoded
                // with 12 sin-only frequency bands per axis (Muller et al. 2021 explicitly
                // omit the cosine half used by NeRF-style encodings). Fed to tcnn as raw
                // Identity dims -- the frequency expansion happens here, not in tcnn.
                Float pn[3] = {
                    (p.x - nrcSceneBounds.pMin.x) / (nrcSceneBounds.pMax.x - nrcSceneBounds.pMin.x),
                    (p.y - nrcSceneBounds.pMin.y) / (nrcSceneBounds.pMax.y - nrcSceneBounds.pMin.y),
                    (p.z - nrcSceneBounds.pMin.z) / (nrcSceneBounds.pMax.z - nrcSceneBounds.pMin.z)};
                constexpr int nPosFreqs = 12;
                for (int axis = 0; axis < 3; ++axis)
                    for (int d = 0; d < nPosFreqs; ++d)
                        row[axis * nPosFreqs + d] = std::sin(Float(1 << d) * pn[axis]);
                // dims 36-37: outgoing direction, spherical (theta,phi) mapped to [0,1] -> OneBlob(4)
                row[36] = std::acos(Clamp(wo.z, -1.f, 1.f)) * InvPi;
                row[37] = (std::atan2(wo.y, wo.x) + Pi) * Inv2Pi;
                // dims 38-39: shading normal, spherical (theta,phi) mapped to [0,1] -> OneBlob(4)
                row[38] = std::acos(Clamp(ns.z, -1.f, 1.f)) * InvPi;
                row[39] = (std::atan2(ns.y, ns.x) + Pi) * Inv2Pi;
                // dim 40: roughness, transformed 1-exp(-r) per Muller et al. -> OneBlob(4)
                row[40] = 1.f - std::exp(-bsdf.Roughness());
                // dims 41-43: diffuse albedo (hemispherical reflectance -> sensor RGB), raw
                RGB albedoRGB = film.ToOutputRGB(albedo, lambda);
                row[41] = float(albedoRGB.r);
                row[42] = float(albedoRGB.g);
                row[43] = float(albedoRGB.b);
                // dims 44-46: specular reflectance F0 (Fresnel at normal incidence), raw.
                // 0 for types with no specular-lobe concept (diffuse, hair, measured, etc.).
                RGB f0RGB(0.f, 0.f, 0.f);
                if constexpr (std::is_same_v<ConcreteBxDF, DielectricBxDF>) {
                    Float f0 = bxdf.F0();
                    f0RGB = RGB(f0, f0, f0);
                } else if constexpr (std::is_same_v<ConcreteBxDF, ConductorBxDF>) {
                    f0RGB = film.ToOutputRGB(bxdf.F0(), lambda);
                }
                row[44] = f0RGB.r;
                row[45] = f0RGB.g;
                row[46] = f0RGB.b;
                // dims 47-48: padding, constant 1 (paper pads to 64 for tile alignment)
                row[47] = 1.f;
                row[48] = 1.f;
                nrcValid[w.pixelIndex] = 1;
            }
#endif


            // Sample light and enqueue shadow ray at intersection point
            BxDFFlags flags = bsdf.Flags();
            if (IsNonSpecular(flags)) {
                // Choose a light source using the _LightSampler_
                LightSampleContext ctx(w.pi, w.n, ns);
                if (IsReflective(flags) && !IsTransmissive(flags))
                    ctx.pi = OffsetRayOrigin(ctx.pi, w.n, wo);
                else if (IsTransmissive(flags) && IsReflective(flags))
                    ctx.pi = OffsetRayOrigin(ctx.pi, w.n, -wo);
                pstd::optional<SampledLight> sampledLight =
                    lightSampler.Sample(ctx, raySamples.direct.uc);
                if (!sampledLight)
                    return;
                Light light = sampledLight->light;

                // Sample light source and evaluate BSDF for direct lighting
                pstd::optional<LightLiSample> ls =
                    light.SampleLi(ctx, raySamples.direct.u, lambda, true);
                if (!ls || !ls->L || ls->pdf == 0)
                    return;
                Vector3f wi = ls->wi;
                SampledSpectrum f = bsdf.f<ConcreteBxDF>(wo, wi);
                if (!f)
                    return;

                // Compute path throughput and path PDFs for light sample
                SampledSpectrum beta = w.beta * f * AbsDot(wi, ns);
                PBRT_DBG("w.beta %f %f %f %f f %f %f %f %f dot %f\n", w.beta[0],
                         w.beta[1], w.beta[2], w.beta[3], f[0], f[1], f[2], f[3],
                         AbsDot(wi, ns));

                PBRT_DBG(
                    "me index %d depth %d beta %f %f %f %f f %f %f %f %f ls.L %f %f %f "
                    "%f ls.pdf %f\n",
                    w.pixelIndex, w.depth, beta[0], beta[1], beta[2], beta[3], f[0], f[1],
                    f[2], f[3], ls->L[0], ls->L[1], ls->L[2], ls->L[3], ls->pdf);

                Float lightPDF = ls->pdf * sampledLight->p;
                // This causes r_u to be zero for the shadow ray, so that
                // part of MIS just becomes a no-op.
                Float bsdfPDF =
                    IsDeltaLight(light.Type()) ? 0.f : bsdf.PDF<ConcreteBxDF>(wo, wi);
                SampledSpectrum r_u = w.r_u * bsdfPDF;
                SampledSpectrum r_l = w.r_u * lightPDF;

                // Enqueue shadow ray with tentative radiance contribution
                SampledSpectrum Ld = beta * ls->L;
                Ray ray = SpawnRayTo(w.pi, w.n, w.time, ls->pLight.pi, ls->pLight.n);
                // Initialize _ray_ medium if media are present
                if (haveMedia)
                    ray.medium = Dot(ray.d, w.n) > 0 ? w.mediumInterface.outside
                                                     : w.mediumInterface.inside;

                shadowRayQueue->Push(ShadowRayWorkItem{ray, 1 - ShadowEpsilon, lambda, Ld,
                                                       r_u, r_l, w.pixelIndex});

                PBRT_DBG("w.index %d spawned shadow ray depth %d Ld %f %f %f %f "
                         "new beta %f %f %f %f beta/uni %f %f %f %f Ld/uni %f %f %f %f\n",
                         w.pixelIndex, w.depth, Ld[0], Ld[1], Ld[2], Ld[3], beta[0],
                         beta[1], beta[2], beta[3], SafeDiv(beta, r_u)[0],
                         SafeDiv(beta, r_u)[1], SafeDiv(beta, r_u)[2],
                         SafeDiv(beta, r_u)[3], SafeDiv(Ld, r_u)[0],
                         SafeDiv(Ld, r_u)[1], SafeDiv(Ld, r_u)[2],
                         SafeDiv(Ld, r_u)[3]);
            }
        });
}

}  // namespace pbrt
