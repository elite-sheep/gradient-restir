
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
import SubpathReuse;
import Rendering.Lights.EmissiveLightSamplerInterface;
import Shift;
import ResamplingCommon;

struct PrefixResampling
{
    ReSTIRPathTracerParams params;
    // Resources
    ReSTIRPT restirpt; 

    Texture2D<PackedHitInfo> vbuffer; 
    Texture2D<PackedHitInfo> temporalVbuffer;


    Texture2D<float2> motionVectors;    
    RWStructuredBuffer<PathReservoir> reservoirs;
    StructuredBuffer<PathReservoir> prevReservoirs;
    //RWTexture2D<float4> vizBuffer;

    RWStructuredBuffer<PrefixGBuffer> prefixGBuffer;
    StructuredBuffer<PrefixGBuffer> prevPrefixGBuffer;

    RWStructuredBuffer<PrefixSearchKey> prefixSearchKeys;

    RWStructuredBuffer<PrefixReservoir> prefixReservoirs;
    StructuredBuffer<PrefixReservoir> prevPrefixReservoirs;
    StructuredBuffer<NeighborValidMask> neighborValidMask;

    Texture1D<float2> neighborOffsets;

    StructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<uint> rcBufferOffsets;

    RWStructuredBuffer<PackedBoundingBox> searchPointBoundingBoxBuffer;
    RWStructuredBuffer<float> prefixTotalLengthBuffer;

    int suffixReuseRoundId;
    float3 prevCameraU;
    float3 prevCameraV;
    float3 prevCameraW; ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float prevJitterX;  ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
    float prevJitterY;  ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).

    float screenSpacePixelSpreadAngle;
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int i)
    {
        if (i == -1) return pixel;

        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * restirpt.subpathSettings.suffixSpatialReuseRadius);

        return neighborPixel;
    }

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */
    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
    {
        int2 pixel = dispatchThreadId.xy;
        const int offset = params.getReservoirOffset(pixel);
        const int pixelId = pixel.y * params.frameDim.x + pixel.x;
        printSetPixel(pixel);
        logSetPixel(pixel);

        bool insideScreen = isValidScreenRegion(params, pixel);
        if (!insideScreen) return;

        // in case of prefix replay, read from the rcDataBuffer
        float3 currentReplayThp = 1.f;
        HitInfo currentPrimaryHit;

        ShadingData currentPrimarySd = getPixelShadingData(params, vbuffer, pixel, currentPrimaryHit);

        if (restirpt.needResetTemporalHistory || !currentPrimaryHit.isValid())
        {
            if (!currentPrimaryHit.isValid())
            {
                searchPointBoundingBoxBuffer[pixelId].minPoint = gScene.camera.data.posW;
                searchPointBoundingBoxBuffer[pixelId].maxPoint = gScene.camera.data.posW;
            }
            prefixGBuffer[offset] = restirpt.prefixGBuffer[offset];
            reservoirs[offset] = restirpt.pathReservoirs[offset];
            return;
        }

        bool isNeighborValid = neighborValidMask[offset].isValid(0) ||
                               !gPathTracer.restirpt.subpathSettings.adaptivePrefixLength;

        SampleGenerator sg = SampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.prefixResamplingSgOffset());
        const PrevCameraFrame pcf = { prevCameraU, prevCameraV, prevCameraW, prevJitterX, prevJitterY };

        PathReservoir subpathReservoir = restirpt.pathReservoirs[offset];

        int numNeighbors = restirpt.subpathSettings.suffixSpatialNeighborCount;
        float2 motionVector = motionVectors[pixel];
        int2 neighborPixel = pixel + motionVector * params.frameDim + 0.5;
        int neighborOffset = params.getReservoirOffset(neighborPixel);
        PrefixGBuffer outputPrefixGBuffer = restirpt.prefixGBuffer[offset];
        PrefixReservoir outputPrefixReservoir = prefixReservoirs[offset];

        // first round
        float3 neighborReplayThp = 1.f;
        PathReservoir neighborReservoir = prevReservoirs[neighborOffset];
        HitInfo neighborPrimaryHit;

        ShadingData neighborPrimarySd = getPixelTemporalShadingData(params, temporalVbuffer, neighborPixel, neighborPrimaryHit, pcf);

        const int startReplayPrefixLength = 1;

        if (!isValidScreenRegion(params, neighborPixel)) neighborPrimaryHit.setInvalid();

        float pathFootprint = 0.f;

        if (restirpt.subpathSettings.useMMIS)
        {
            pathFootprint = outputPrefixReservoir.pathFootprint;
        }
        else
        {
            if (gPathTracer.restirpt.subpathSettings.adaptivePrefixLength && subpathReservoir.pathFlags.prefixLength() > startReplayPrefixLength)
            {
                neighborReplayThp = reconnectionDataBuffer[rcBufferOffsets[2 * offset]].pathThroughput;

                neighborPrimarySd = loadShadingData(reconnectionDataBuffer[rcBufferOffsets[2 * offset]].rcPrevHit, float3(0.f),
                                                    -reconnectionDataBuffer[rcBufferOffsets[2 * offset]].rcPrevWo, false,
                                                    ExplicitLodTextureSampler(params.lodBias), true);

            }

            float prefixTotalLengthBeforeRc = 0.f;
            if (gPathTracer.restirpt.subpathSettings.adaptivePrefixLength && neighborReservoir.pathFlags.prefixLength() > startReplayPrefixLength)
            {
                currentReplayThp = reconnectionDataBuffer[rcBufferOffsets[2 * offset + 1]].pathThroughput;

                currentPrimarySd = loadShadingData(reconnectionDataBuffer[rcBufferOffsets[2 * offset + 1]].rcPrevHit, float3(0.f),
                                                    -reconnectionDataBuffer[rcBufferOffsets[2 * offset + 1]].rcPrevWo, false, 
                                                    ExplicitLodTextureSampler(params.lodBias), false);

                prefixTotalLengthBeforeRc = prefixTotalLengthBuffer[pixelId];
            }
            else
            {
                prefixTotalLengthBeforeRc = length(currentPrimarySd.posW - gScene.camera.data.posW);
            }

            pathFootprint = prefixTotalLengthBeforeRc;

            ResamplePrefix(isNeighborValid, subpathReservoir,
                            neighborReservoir, restirpt.subpathSettings.temporalHistoryLength,
                            currentPrimaryHit, currentPrimarySd, neighborPrimaryHit, neighborPrimarySd,
                            currentReplayThp, neighborReplayThp,
                            outputPrefixGBuffer, outputPrefixReservoir,
                            prevPrefixGBuffer[neighborOffset],
                            prevPrefixReservoirs[neighborOffset],
                            sg, (kMaxSurfaceBounces + 1) * offset, pathFootprint);
            prefixReservoirs[offset] = outputPrefixReservoir;
        }
        
        prefixGBuffer[offset] = outputPrefixGBuffer;
        reservoirs[offset] = subpathReservoir;
        
        uint pointIndex = pixelId;

        float searchRadius;

        if (gPathTracer.restirpt.subpathSettings.knnSearchAdaptiveRadiusType != (uint)ReSTIRPT::KNNAdaptiveRadiusType::NonAdaptive)
        {
            float minRes = min(params.frameDim.x, params.frameDim.y);
            searchRadius = restirpt.subpathSettings.knnSearchRadiusMultiplier * pathFootprint * screenSpacePixelSpreadAngle;// / (minRes * minRes);
        }
        else
        {
            // mitsuba's photon mapping initial guess
            searchRadius = restirpt.subpathSettings.knnSearchRadiusMultiplier * min(restirpt.sceneRadius / params.frameDim.x, restirpt.sceneRadius / params.frameDim.y);
        }


        float3 hitPosition = outputPrefixGBuffer.hit.isValid() ? loadVertexPosition(outputPrefixGBuffer.hit) : gScene.camera.data.posW;
        searchRadius = outputPrefixGBuffer.hit.isValid() ? searchRadius : 0.f;

        // write bounding box buffer
        searchPointBoundingBoxBuffer[pointIndex].minPoint = hitPosition - searchRadius;
        searchPointBoundingBoxBuffer[pointIndex].maxPoint = hitPosition + searchRadius;

        if (gPathTracer.restirpt.subpathSettings.knnIncludeDirectionSearch)
        {
            bool isPrefixValid = outputPrefixGBuffer.hit.isValid();
            prefixSearchKeys[offset].wo = isPrefixValid ? outputPrefixGBuffer.wo : 0.f;
        }
    }
}

cbuffer CB
{
    PrefixResampling gPrefixResampling;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
{
    gPrefixResampling.run(dispatchThreadId, groupThreadIdx, groupID);
}
