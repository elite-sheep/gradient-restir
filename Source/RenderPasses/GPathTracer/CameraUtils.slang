// Copyright @yucwang 2024

import Scene.Camera.CameraData;
__exported import Utils.Math.Ray;
import Utils.Math.MathHelpers;

Ray computeRayPinhole(const CameraData data,
                      float2 pixel,
                      uint2 frameDim)
{
    Ray ray;

    // Compute the normalized ray direction assuming a pinhole camera.
    ray.origin = data.posW;
    ray.dir = normalize(computeNonNormalizedRayDirPinhole(data,
                                                          pixel,
                                                          frameDim));

    float invCos = 1.f / dot(normalize(data.cameraW), ray.dir);
    ray.tMin = data.nearZ * invCos;
    ray.tMax = data.farZ * invCos;

    return ray;
}

float3 computeNonNormalizedRayDirPinhole(const CameraData data,
                                         float2 pixel,
                                         uint2 frameDim)
{
    float2 p = pixel / frameDim;
    float2 ndc = float2(2, -2) * p + float2(-1, 1);

    // Compute the non-normalized ray direction assuming a pinhole camera.
    return ndc.x * data.cameraU + ndc.y * data.cameraV + data.cameraW;
}

float computePinholeReconnectionDifferential( const CameraData data,
                                              float2 fracPixelLoc,
                                              float3 primaryPosW,
                                              float3 primaryN,
                                              uint2 frameDim )
{
    // float2 pixelPosScreen = fracPixelLoc / frameDim;
    // float2 pixelPosNdc = float2(2, -2) * pixelPosScreen + float2(-1, 1);
    // float3 pixelPosW = data.posW + data.cameraW + (pixelPosNdc.x * data.cameraU + pixelPosNdc.y * data.cameraV);

    // float pToPixel = length(pixelPosW - data.posW);        // d0
    // float pixelToPrimary = length(primaryPosW - pixelPosW);// d1

    // float3 rayDir = normalizeSafe(pixelPosNdc.x * data.cameraU + pixelPosNdc.y * data.cameraV + data.cameraW);
    // float cosineTerm0 = abs(dot(normalizeSafe(data.cameraW), rayDir));
    // float cosineTerm1 = abs(dot(normalizeSafe(primaryN), rayDir));

    // float res = sqr(pToPrimary) * sqr(cosineTerm0) * cosineTerm0 / cosineTerm1;
    // if (isnan(res) || isinf(res))
    // {
    //     res = 0.0f;
    // }

    // return res;
    return 1.0f;
}
