/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Params;
//import Shift;
import Scene.HitInfo;
//import PathReservoir;
//import PathTracer;
import Scene.Scene;
import Utils.Debug.PixelDebug;
//import Rendering.Utils.PixelStats;
//import Utils.Sampling.TinyUniformSampleGenerator;
//import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import LoadShadingData;
// import HybridShiftTraceRays;
import ReSTIRPT;


bool isValidPackedHitInfo(PackedHitInfo packed)
{
    return packed.x != 0;
}


bool isPrimaryHitValid(Texture2D<PackedHitInfo> vbuffer, int2 pixel, int offset)
{
    HitInfo hit = HitInfo(vbuffer[pixel]);
    return hit.isValid();
}


bool isValidScreenRegion(ReSTIRPathTracerParams params, int2 pixel) { return all(pixel >= 0 && pixel < params.frameDim); }

bool isValidGeometry(ShadingData centralSd, ShadingData neighborSd)
{
    // if (!gFeatureBasedRejection) return true;
    float3 camPos = gScene.camera.data.posW;
    float centralDist = distance(camPos, centralSd.posW);
    float neighborDist = distance(camPos, neighborSd.posW);
    return dot(centralSd.frame.N, neighborSd.frame.N) >= 0.5f && abs(centralDist - neighborDist) < 0.1f * centralDist;
}


HitInfo getPixelHitInfo(ReSTIRPathTracerParams params, Texture2D<PackedHitInfo> vbuffer, int2 pixel, int offset, out float3 V)

{
    Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);
    V = -ray.dir;

    return HitInfo(vbuffer[pixel]);
    
}

ShadingData getPixelShadingData(ReSTIRPathTracerParams params, Texture2D<PackedHitInfo> vbuffer, int2 pixel, out HitInfo hit)
{
    ShadingData sd = {};
    Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);
    hit = HitInfo(vbuffer[pixel]);
    if (hit.isValid())
    {
        sd = loadShadingData(hit, ray.origin, ray.dir, true, ExplicitLodTextureSampler(0), false);
    }
    return sd;
}

// doesn't pass HitInfo
ShadingData getPixelShadingData(ReSTIRPathTracerParams params, Texture2D<PackedHitInfo> vbuffer, int2 pixel)
{
    ShadingData sd = {};
    Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);
    HitInfo hit = HitInfo(vbuffer[pixel]);
    if (hit.isValid())
    {
        sd = loadShadingData(hit, ray.origin, ray.dir, true, ExplicitLodTextureSampler(0), false);
    }
    return sd;
}

struct PrevCameraFrame
{
    float3 prevCameraU;
    float3 prevCameraV;
    float3 prevCameraW;
    float prevJitterX;
    float prevJitterY;
}

Ray computeRayPinholePrevFrame(const Camera camera, uint2 pixel, uint2 frameDim, const PrevCameraFrame pcf)
{
    // can we get a camera ray in the previous frame?
    Ray ray;

    // Compute the normalized ray direction assuming a pinhole camera.
    ray.origin = camera.data.prevPosW;

    // Compute sample position in screen space in [0,1] with origin at the top-left corner.
    // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
    float2 p = (pixel + float2(0.5f, 0.5f)) / frameDim + float2(-pcf.prevJitterX, pcf.prevJitterY);
    float2 ndc = float2(2, -2) * p + float2(-1, 1);

    // Compute the non-normalized ray direction assuming a pinhole camera.
    ray.dir = normalize(ndc.x * pcf.prevCameraU + ndc.y * pcf.prevCameraV + pcf.prevCameraW);

    float invCos = 1.f / dot(normalize(pcf.prevCameraW), ray.dir);
    ray.tMin = camera.data.nearZ * invCos;
    ray.tMax = camera.data.farZ * invCos;

    return ray;
}


HitInfo getPixelTemporalHitInfo(ReSTIRPathTracerParams params, Texture2D<PackedHitInfo> temporalVbuffer, int2 pixel, int offset, out float3 V, const PrevCameraFrame pcf)
                                
{
    Ray ray = computeRayPinholePrevFrame(gScene.camera, pixel, params.frameDim, pcf);
    V = -ray.dir;

    return HitInfo(temporalVbuffer[pixel]);

}

ShadingData getPixelTemporalShadingData(ReSTIRPathTracerParams params, Texture2D<PackedHitInfo> temporalVbuffer, int2 pixel, out HitInfo hit, const PrevCameraFrame pcf)
{
    ShadingData sd = {};
    Ray ray = computeRayPinholePrevFrame(gScene.camera, pixel, params.frameDim, pcf);
    hit = HitInfo(temporalVbuffer[pixel]);
    if (hit.isValid())
    {
        sd = loadShadingData(hit, ray.origin, ray.dir, true, ExplicitLodTextureSampler(0), true);
    }
    return sd;
}

ShadingData getPixelShadingDataGeneral(ReSTIRPathTracerParams params, Texture2D<PackedHitInfo> vbuffer, Texture2D<PackedHitInfo> temporalVbuffer,
                                       int2 pixel, out HitInfo hit, bool isTemporal, const PrevCameraFrame pcf)
{
    ShadingData sd = {};
    Ray ray;
    if (isTemporal)
        ray = computeRayPinholePrevFrame(gScene.camera, pixel, params.frameDim, pcf);
    else
        ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);

    if (isTemporal)
        hit = HitInfo(temporalVbuffer[pixel]);
    else
        hit = HitInfo(vbuffer[pixel]);

    if (hit.isValid())
    {
        sd = loadShadingData(hit, ray.origin, ray.dir, true, ExplicitLodTextureSampler(0), isTemporal);
    }

    return sd;
}

// void unpackSearchedPixels(int2 reprojPixel, int packedData, inout int2 outPixels[2])
// {
//     int2 offset0;
//     int2 offset1;
//     offset0.x = (packedData << 24) >> 24;
//     offset0.y = (packedData << 16) >> 24;
//     offset1.x = (packedData << 8) >> 24;
//     offset1.y = packedData >> 24;
//     outPixels[0] = reprojPixel + offset0;
//     outPixels[1] = reprojPixel + offset1;
// }

// int2 unpackSearchedPixel(int2 reprojPixel, int packedData, int index)
// {
//     int basis = index == 0 ? 16 : 0;
//     int2 offset;
//     offset.x = (packedData << (8 + basis)) >> 24;
//     offset.y = (packedData << basis) >> 24;
//     return reprojPixel + offset;
// }

int2 unpackSearchedPixel(int packedData)
{
    return int2(packedData & 0xFFFF, packedData >> 16 & 0xFFFF); 
}

