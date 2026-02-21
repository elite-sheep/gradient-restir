// Copyright @yucwang 2024

import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGenerator;

float SafeNumber(float a)
{
    if (isnan(a) || isinf(a))
    {
        return 0.0f;
    }
    else
    {
        return a;
    }
}

void computeBilinearWeights(float2 fracPixelLocMin, inout float4 bilinearWeights)
{
    bilinearWeights.x = (1.0f - fracPixelLocMin.x) * (1.0f - fracPixelLocMin.y);
    bilinearWeights.y = fracPixelLocMin.x * (1.0f - fracPixelLocMin.y);
    bilinearWeights.z = (1.0f - fracPixelLocMin.x) * fracPixelLocMin.y;
    bilinearWeights.w = fracPixelLocMin.x * fracPixelLocMin.y;
}

bool isJacobianInvalid(float Jacobian)
{
    return Jacobian <= 0.f || isnan(Jacobian) || isinf(Jacobian);
}

bool isIntegrandInvalid(float3 Integrand)
{
    return all(Integrand == 0.f) || any(isnan(Integrand) || isinf(Integrand) || Integrand < 0.f);
}

inline uint setBit(uint i, int iBit)
{
    return i | (uint)(1 << iBit);
}

inline uint clearBit(uint i, int iBit)
{
    return i & (uint) ~(1 << iBit);
}

inline bool getBit(uint i, int iBit)
{
    return (i & (uint) (1 << iBit)) > 0;
}

inline uint getBufferOffset(uint2 pixel, uint2 frameDim)
{
    return frameDim.y * pixel.x + pixel.y;
}

inline uint getBufferOffset(uint2 pixel, uint2 frameDim, uint pathsPerPixel)
{
    uint pixelID = getBufferOffset(pixel, frameDim);
    return pathsPerPixel * pixelID;
}

inline bool isInFrameFloat(float2 pixel, uint2 frameDim)
{
    return all(pixel >= 0.0f) && all(pixel < float2(frameDim));
}

inline bool isInFrame(int2 pixel, uint2 frameDim)
{
    return all(pixel >= 0) && all(pixel < frameDim);
}

inline bool isInPixel(float2 fracPixelLoc, uint2 pixel)
{
    return all(fracPixelLoc > float2(pixel)) && all(fracPixelLoc < float2(pixel + uint2(1, 1)));
}

inline float posTargetFunc(float3 color)
{
    return targetFunc(max(color, float3(0.0f)));
    // return targetFunc(abs(color));
    // return max(targetFunc(color), 0.0f);
}

inline float negTargetFunc(float3 color)
{
    return targetFunc(max(-color, float3(0.0f)));
    // return max(-targetFunc(color), 0.0f);
    // return 0.0f;
}

inline float avg(float3 rgb)
{
    return (rgb.x + rgb.y + rgb.z) / 3.0f;
}

inline float maxElement(float3 rgb)
{
    return max(rgb.x, max(rgb.y, rgb.z));
}

inline float targetFunc(float3 color)
{
    return luminance(color);
    // return avg(color);
}

inline bool shouldReuseBasedOnJacobian(float j1, float j2)
{
    const float epsilon = 1e-6;
    if ((j1 < epsilon && j2 > epsilon) || (j1 > epsilon && j2 < epsilon))
        return false;
    else if (j1 < epsilon && j2 < epsilon)
        return true;
    else
    {
        float jRatio = min(j1 / j2, j2 / j1);
        if (jRatio > 0.1f)
            return true;
    }
    return false;
}

inline uint float2ToUint(float2 v)
{
    uint a = (uint) round(v.x * 65535.0f);
    uint b = (uint) round(v.y * 65535.0f);
    a = clamp(a, 0, 65535);
    b = clamp(b, 0, 65535);
    return a | (b << 16);
}

inline float2 uintToFloat2(uint v)
{
    static const uint mask = 0x0000FFFF;
    return float2(clamp(v & mask, 0, mask), clamp((v >> 16) & mask, 0, mask)) / 65535.0f;
}

inline uint2 float2ToUint2(float2 v)
{
    uint a = (uint) round(v.x * 1048575.0f);
    uint b = (uint) round(v.y * 1048575.0f);
    a = clamp(a, 0, 1048575);
    b = clamp(b, 0, 1048575);
    return uint2(a, b);
}

inline float2 uint2ToFloat2(uint2 v)
{
    static const uint mask = 0x000FFFFF;
    return float2(clamp(v.x & mask, 0, mask), clamp(v.y & mask, 0, mask)) / 1048575.0f;
}

inline uint popcount32(uint x)
{
    x = x - ((x >> 1) & 0x55555555);               // put count of each 2 bits in 2 bits
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);// 4‑bit partial sums
    x = (x + (x >> 4)) & 0x0F0F0F0F;               // 8‑bit partial sums
    x = x + (x >> 8);                              // 16‑bit sums
    x = x + (x >> 16);                             // full 32‑bit sum
    return x & 0x3F;                               // 0–32 fits in 6 bits
}

void consumeRng(inout SampleGenerator sg, uint sampleCount)
{
    for (int i = 0; i < sampleCount; ++i)
        sampleNext1D(sg);
}
