
/***************************************************************************
# Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
**************************************************************************/
#include "Utils/Math/MathConstants.slangh"
import Scene.RaytracingInline;
import ReSTIRPT;
import Scene.HitInfo;
import PathReservoir;
import PathTracer;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Color.ColorHelpers;
import LoadShadingData;
import Params;
import ResamplingCommon;

/** Simple scheduler (fullscreen compute pass) using TraceRayInline.
 */
struct PrefixNeighborSearch
{
    ReSTIRPathTracerParams params; ///< Runtime parameters.
    // Resources
    ReSTIRPT restirpt;                ///< Settings for ReSTIR PT. Only valid if kReSTIRPT == true.
    Texture2D<float2> motionVectors;    

    StructuredBuffer<PrefixGBuffer> prefixGBuffer;
    StructuredBuffer<PrefixGBuffer> prevPrefixGBuffer;
    StructuredBuffer<PrefixSearchKey> prefixSearchKeys;

    Texture1D<float2> neighborOffsets;

    RWStructuredBuffer<uint> temporalNeighborPixels;

    RWStructuredBuffer<PackedBoundingBox> searchPointBoundingBoxBuffer;

    int integrationPrefixId;

    RaytracingAccelerationStructure gSearchPointAS;

    bool hasSearchPointAS;

    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int radius, int i)
    {
        if (i == -1) return pixel;

        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * radius);

        return neighborPixel;
    }

    bool isPixelPrefixLastSdValid(int offset)
    {
        PrefixGBuffer pg = prefixGBuffer[offset];
        return pg.hit.isValid();
    }

    void run(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
    {
        int2 pixel = dispatchThreadId.xy;
        const int offset = params.getReservoirOffset(pixel);
        // fetch the V Buffer

        printSetPixel(pixel);
        logSetPixel(pixel);

        bool insideScreen = isValidScreenRegion(params, pixel);
        if (!insideScreen) return;
        bool centralPrefixValid = isPixelPrefixLastSdValid(offset);
        if (!centralPrefixValid) return;

        SampleGenerator sg = SampleGenerator(pixel, restirpt.sgCount() * params.seed 
        + restirpt.prefixNeighborSearchOffset(integrationPrefixId));

        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

        int numNeighbors = restirpt.subpathSettings.finalGatherSuffixCount;

        float2 motionVector = motionVectors[pixel];
        int2 reprojPixel = pixel;
        float canonicalMisWeights = 0.f;

        float3 currentPrefixDir = prefixGBuffer[offset].wo; // this is the scratch prefix g buffer
        float3 currentPrefixPos = loadVertexPosition(prefixGBuffer[offset].hit);

        float scores[8];
        uint pixels[8];
        float currentMin = -1e10f;
        int currentMinId = 0;
        for (int i=0; i < numNeighbors; i++)
        {
            scores[i] = -1e10f;
            pixels[i] = 0;
        }

        int sr = restirpt.subpathSettings.prefixNeighborSearchRadius;

        int searchNeighborCount = restirpt.subpathSettings.prefixNeighborSearchNeighborCount;

        // nearest neighbor search
        // if the integration prefix is too glossy, we use screen space search to find a neighbor with directional criterion
        bool needDirSample = restirpt.subpathSettings.knnIncludeDirectionSearch && numNeighbors > 1;

        int knnNeighbors = needDirSample ? numNeighbors - 1 : numNeighbors;

        if (hasSearchPointAS)
        {
            float searchRadius = restirpt.subpathSettings.knnSearchRadiusMultiplier * min(restirpt.sceneRadius / params.frameDim.x, restirpt.sceneRadius / params.frameDim.y);

            RayDesc searchRay;
            searchRay.Origin = currentPrefixPos;
            searchRay.Direction = float3(0.0f, 1.0f, 0.0f);
            searchRay.TMin = 0.0f;
            searchRay.TMax = 0.0f;
            RayQuery<RAY_FLAG_NONE> rayQuery;
            rayQuery.TraceRayInline(gSearchPointAS, RAY_FLAG_NONE, 0xff, searchRay);
            
            while (rayQuery.Proceed())
            {
                if (rayQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
                {
                    uint photonIndex = rayQuery.CandidatePrimitiveIndex();

                    PackedBoundingBox aabb = searchPointBoundingBoxBuffer[photonIndex];
                    float3 photonPosition = 0.5 * (aabb.maxPoint + aabb.minPoint);
                    float searchRadius2 = 0.25 * dot(aabb.maxPoint - aabb.minPoint, aabb.maxPoint - aabb.minPoint);

                    float disSquared = dot(photonPosition - currentPrefixPos, photonPosition - currentPrefixPos);

                    // some how we want to avoid using self
                    if (disSquared == 0) continue;

                    if (disSquared < searchRadius2)
                    {
                        float curScore = -disSquared;

                        if (curScore > currentMin) 
                        {
                            scores[currentMinId] = curScore;
                            pixels[currentMinId] = (photonIndex % params.frameDim.x) | (photonIndex / params.frameDim.x) << 16;
                            currentMin = 1e10f;
                            for (int i = 0; i < knnNeighbors; i++)
                            {
                                if (scores[i] < currentMin)
                                {
                                    currentMin = scores[i];
                                    currentMinId = i;
                                }
                            }
                        }
                    }
                }

            }
        }

        if (numNeighbors > 1 && needDirSample)
        {
            // lock the first numNeighbors - 1 neighbors (if exits), clean the last neighbor
            currentMinId = numNeighbors-1;
            scores[numNeighbors-1] = -1e10f;
            currentMin = -1e10f;

            for (int i = 0; i < knnNeighbors; i++)
            {
                if (scores[i] != -1e10f)                        
                    scores[i] = 1e10f;
                else 
                {
                    currentMinId = i;
                    break;
                } 
            }

            for (int neighborId = 0; neighborId < searchNeighborCount; neighborId++)
            {
                int2 neighborPixel = getNextNeighborPixel(startIndex, reprojPixel, sr, neighborId);
                HitInfo temporalPrimaryHit;
                if (!isValidScreenRegion(params, neighborPixel)) continue;

                int neighborOffset = params.getReservoirOffset(neighborPixel);

                PrefixSearchKey pg = prefixSearchKeys[neighborOffset];
                float curDot = dot(currentPrefixDir, pg.wo);

                if (curDot > currentMin)
                {
                    scores[currentMinId] = curDot;
                    pixels[currentMinId] = neighborPixel.x | neighborPixel.y << 16;
                    // search for current Min
                    currentMin = 1e10f;
                    for (int i = 0; i < numNeighbors; i++)
                    {
                        if (scores[i] < currentMin)
                        {
                            currentMin = scores[i];
                            currentMinId = i;
                        }
                    }
                }
            }
        }

        // can only use one temporal neighbor anyway

        // must be from [-128,127]
        for (int i = 0; i < numNeighbors; i++)
        {
            if (scores[i] == -1e10f)
            {
                int2 pixel_ = getNextNeighborPixel(startIndex, reprojPixel, restirpt.subpathSettings.suffixSpatialReuseRadius, i);
                temporalNeighborPixels[numNeighbors * offset + i] = (pixel_.x & 0xFFFF) | (pixel_.y & 0xFFFF) << 16;
            }
            else
            {
                temporalNeighborPixels[numNeighbors * offset + i] = pixels[i];
            }
        }
    }
}

cbuffer CB
{
    PrefixNeighborSearch gPrefixNeighborSearch;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
{
    gPrefixNeighborSearch.run(dispatchThreadId, groupThreadIdx, groupID);
}
