#ifndef PBRT_NRC_ENCODING_H
#define PBRT_NRC_ENCODING_H

#include <pbrt/pbrt.h>

#include <cmath>
#include <cstdint>

namespace pbrt
{
    namespace nrc
    {

        static constexpr uint32_t kNRCPositionDims = 36; // 3 coordinates * 12 waves
        static constexpr uint32_t kNRCDirectionDims = 8; // 2 spherical coords * 4 blobs
        static constexpr uint32_t kNRCNormalDims = 8;    // 2 spherical coords * 4 blobs
        static constexpr uint32_t kNRCRoughnessDims = 4; // 1 scalar * 4 blobs
        static constexpr uint32_t kNRCDiffuseDims = 3;   // RGB
        static constexpr uint32_t kNRCSpecularDims = 3;  // RGB
        static constexpr uint32_t kNRCPaddingDims = 2;

        static constexpr uint32_t kNRCInputDims =
            kNRCPositionDims +
            kNRCDirectionDims +
            kNRCNormalDims +
            kNRCRoughnessDims +
            kNRCDiffuseDims +
            kNRCSpecularDims +
            kNRCPaddingDims; // 64

        static_assert(kNRCInputDims == 64, "NRC input must be 64 dimensions.");

        PBRT_CPU_GPU inline float NRCClamp(float x, float lo, float hi)
        {
            return x < lo ? lo : (x > hi ? hi : x);
        }

        PBRT_CPU_GPU inline float NRCSafeInv(float x)
        {
            return std::abs(x) > 1e-8f ? 1.0f / x : 0.0f;
        }

        PBRT_CPU_GPU inline float NRCFract(float x)
        {
            return x - std::floor(x);
        }

        // Triangle wave approximation used instead of sine-like frequency encoding.
        // Output range is approximately [-1, 1].
        PBRT_CPU_GPU inline float NRCTriangleWave(float x)
        {
            float m = x - 2.0f * std::floor(x * 0.5f); // x mod 2, positive for x >= 0
            return 2.0f * std::abs(m - 1.0f) - 1.0f;
        }

        // Quartic blob kernel.
        // Input d is distance in "blob index space".
        // Nonzero only for |d| < 1.
        PBRT_CPU_GPU inline float NRCQuarticBlob(float d)
        {
            d = std::abs(d);
            if (d >= 1.0f)
                return 0.0f;

            float t = 1.0f - d * d;
            return (15.0f / 16.0f) * t * t;
        }

        // One-blob encoding with 4 evenly spaced centers.
        // v should be in [0, 1].
        // out must point to 4 floats.
        PBRT_CPU_GPU inline void NRCOneBlob4(float v, float *out)
        {
            v = NRCClamp(v, 0.0f, 1.0f);

            // Map [0,1] to blob-index space [0,3].
            float x = v * 3.0f;

            out[0] = NRCQuarticBlob(x - 0.0f);
            out[1] = NRCQuarticBlob(x - 1.0f);
            out[2] = NRCQuarticBlob(x - 2.0f);
            out[3] = NRCQuarticBlob(x - 3.0f);
        }

        PBRT_CPU_GPU inline void NRCNormalize3(float x, float y, float z,
                                               float *ox, float *oy, float *oz)
        {
            float len2 = x * x + y * y + z * z;

            if (len2 <= 1e-12f)
            {
                *ox = 0.0f;
                *oy = 0.0f;
                *oz = 1.0f;
                return;
            }

            float invLen = 1.0f / std::sqrt(len2);
            *ox = x * invLen;
            *oy = y * invLen;
            *oz = z * invLen;
        }

        // Convert a 3D direction to two spherical coordinates in [0,1].
        // theta01: polar angle, 0 at +z, 1 at -z
        // phi01: azimuth angle around z
        PBRT_CPU_GPU inline void NRCDirectionToSpherical01(float x, float y, float z,
                                                           float *theta01,
                                                           float *phi01)
        {
            constexpr float Pi = 3.14159265358979323846f;
            constexpr float InvPi = 1.0f / Pi;
            constexpr float InvTwoPi = 1.0f / (2.0f * Pi);

            float nx, ny, nz;
            NRCNormalize3(x, y, z, &nx, &ny, &nz);

            nz = NRCClamp(nz, -1.0f, 1.0f);

            float theta = std::acos(nz);
            float phi = std::atan2(ny, nx);

            if (phi < 0.0f)
                phi += 2.0f * Pi;

            *theta01 = theta * InvPi;
            *phi01 = phi * InvTwoPi;
        }

        // Position encoding.
        // px, py, pz should already be normalized to [0,1] using scene bounds.
        // out must point to 36 floats.
        PBRT_CPU_GPU inline void NRCEncodePosition(float px, float py, float pz,
                                                   float *out)
        {
            px = NRCClamp(px, 0.0f, 1.0f);
            py = NRCClamp(py, 0.0f, 1.0f);
            pz = NRCClamp(pz, 0.0f, 1.0f);

            float p[3] = {px, py, pz};

            int index = 0;
            for (int c = 0; c < 3; ++c)
            {
                float frequency = 1.0f;

                for (int f = 0; f < 12; ++f)
                {
                    out[index++] = NRCTriangleWave(p[c] * frequency);
                    frequency *= 2.0f;
                }
            }
        }

        // Direction or normal encoding.
        // Input is a 3D vector.
        // Output is 8 floats: one-blob(theta) + one-blob(phi).
        PBRT_CPU_GPU inline void NRCEncodeDirectionLike(float x, float y, float z,
                                                        float *out)
        {
            float theta01, phi01;
            NRCDirectionToSpherical01(x, y, z, &theta01, &phi01);

            NRCOneBlob4(theta01, out + 0);
            NRCOneBlob4(phi01, out + 4);
        }

        // Roughness encoding.
        // Müller uses 1 - exp(-roughness) before one-blob encoding.
        // out must point to 4 floats.
        PBRT_CPU_GPU inline void NRCEncodeRoughness(float roughness, float *out)
        {
            roughness = roughness < 0.0f ? 0.0f : roughness;

            float v = 1.0f - std::exp(-roughness);
            NRCOneBlob4(v, out);
        }

        // Main NRC input encoder.
        // Position must be normalized before calling this function.
        PBRT_CPU_GPU inline void EncodeNRCInput64(
            float px, float py, float pz,
            float woX, float woY, float woZ,
            float nX, float nY, float nZ,
            float roughness,
            float diffuseR, float diffuseG, float diffuseB,
            float specularR, float specularG, float specularB,
            float *out)
        {
            int offset = 0;

            NRCEncodePosition(px, py, pz, out + offset);
            offset += kNRCPositionDims;

            NRCEncodeDirectionLike(woX, woY, woZ, out + offset);
            offset += kNRCDirectionDims;

            NRCEncodeDirectionLike(nX, nY, nZ, out + offset);
            offset += kNRCNormalDims;

            NRCEncodeRoughness(roughness, out + offset);
            offset += kNRCRoughnessDims;

            out[offset++] = diffuseR;
            out[offset++] = diffuseG;
            out[offset++] = diffuseB;

            out[offset++] = specularR;
            out[offset++] = specularG;
            out[offset++] = specularB;

            // Padding. Müller pads to 64 with value 1, which also acts like a bias input.
            out[offset++] = 1.0f;
            out[offset++] = 1.0f;
        }

        // Helper for converting world position to normalized [0,1] position.
        PBRT_CPU_GPU inline void NRCNormalizePositionToBounds(
            float x, float y, float z,
            float minX, float minY, float minZ,
            float maxX, float maxY, float maxZ,
            float *px, float *py, float *pz)
        {
            float dx = maxX - minX;
            float dy = maxY - minY;
            float dz = maxZ - minZ;

            *px = dx > 1e-8f ? (x - minX) / dx : 0.5f;
            *py = dy > 1e-8f ? (y - minY) / dy : 0.5f;
            *pz = dz > 1e-8f ? (z - minZ) / dz : 0.5f;

            *px = NRCClamp(*px, 0.0f, 1.0f);
            *py = NRCClamp(*py, 0.0f, 1.0f);
            *pz = NRCClamp(*pz, 0.0f, 1.0f);
        }

    } // namespace nrc
} // namespace pbrt

#endif // PBRT_NRC_ENCODING_H