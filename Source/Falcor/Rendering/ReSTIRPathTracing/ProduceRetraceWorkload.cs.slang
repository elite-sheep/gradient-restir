/***************************************************************************
# Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
**************************************************************************/
import Utils.Attributes;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;
import ReSTIRPT;
import LoadShadingData;
import RetraceWorkloadQueue;
import Params;
import Scene.HitInfo;
import PathReservoir;
import Utils.Sampling.TinyUniformSampleGenerator;
import ResamplingCommon;
import Rendering.Utils.PixelStats;


// Shared memory variables.
static const uint kWarpCount = (kScreenTileDim.x * kScreenTileDim.y) / 32;
groupshared uint gWorkloadOffset[8 /* kWarpCount */];

struct PathGenerator
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    ReSTIRPathTracerParams params;
    // Resources
    ReSTIRPT restirpt; ///< Settings for ReSTIR PT. Only valid if kReSTIRPT == true.


    Texture2D<PackedHitInfo> vbuffer; ///< Fullscreen V-buffer for the primary hits.
    Texture2D<PackedHitInfo> temporalVbuffer;


    Texture1D<float2> neighborOffsets;
    Texture2D<float2> motionVectors;

    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<PathReservoir> reservoirs;
    RWStructuredBuffer<NeighborValidMask> neighborValidMask;

    float3 prevCameraU;
    float3 prevCameraV;
    float3 prevCameraW; ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float prevJitterX;  ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
    float prevJitterY;  ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).
    bool gEnableTemporalReprojection;
    bool gNoResamplingForTemporalReuse;
    uint gTemporalHistoryLength;

    int gSpatialRoundId;
    int gNumSpatialRounds;
    // for default pattern
    int gNeighborCount;
    float gGatherRadius;

    bool isSpatialPass;
    bool gAdditionalSpatial;

    RetraceWorkloadQueue queue;                   ///< Work queues we append paths to.

    int getNeighborCount()
    {
        return isSpatialPass ? gNeighborCount : 1; // does not include self
    }

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int i)
    {
        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * gGatherRadius);

        return neighborPixel;
    }

    /** Entry point for path generator.
        \param[in] tileID Tile ID in x and y on screen.
        \param[in] threadIdx Thread index within the tile.
    */
    void execute(const uint2 tileID, const uint threadIdx)
    {
        const uint2 tileOffset = tileID << kScreenTileBits;            // Tile offset in pixels.
        const uint2 pixel = deinterleave_8bit(threadIdx) + tileOffset; // Assumes 16x16 tile or smaller.
        const PrevCameraFrame pcf = { prevCameraU, prevCameraV, prevCameraW, prevJitterX, prevJitterY };

        uint numWorks = 0;
        uint workMask = 0;

        int neighborCount = getNeighborCount();
        if (all(pixel < params.frameDim))
        {
            // figure out neighbors
            var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed +
                                                           restirpt.spatialOrTemporalSgOffset(isSpatialPass, gAdditionalSpatial, gSpatialRoundId));

            const uint centralOffset = params.getReservoirOffset(pixel);

            float3 centralV;
            HitInfo centralPrimaryHit;
            
            ShadingData centralPrimarySd = getPixelShadingData(params, vbuffer, pixel, centralPrimaryHit);
            
            if (centralPrimaryHit.isValid())
            {
                if (isSpatialPass)
                {
                    PathReservoir centralReservoir = prevReservoirs[centralOffset];

                    const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

                    for (int i = 0; i < neighborCount; ++i)
                    {
                        int2 neighborPixel = getNextNeighborPixel(startIndex, pixel, i);
                        int neighborOffset = params.getReservoirOffset(neighborPixel);

                        if (!isValidScreenRegion(params, neighborPixel))
                        {
                            neighborValidMask[centralOffset].setValid(i, false);
                            continue;
                        }

                        PathReservoir neighborReservoir = prevReservoirs[neighborOffset];


                        HitInfo neighborPrimaryHit;
                        ShadingData neighborPrimarySd = getPixelShadingData(params, vbuffer, neighborPixel, neighborPrimaryHit);
                        if (!neighborPrimaryHit.isValid() || !isValidGeometry(centralPrimarySd, neighborPrimarySd))

                        {
                            neighborValidMask[centralOffset].setValid(i, false);
                            continue;
                        }

                        neighborValidMask[centralOffset].setValid(i, true);

                        if (ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
                        {
                            int p1 = int(centralReservoir.pathFlags.rcVertexLength() > 1 && centralReservoir.pathFlags.pathLength() > 0);
                            numWorks += p1;
                            workMask |= p1 << (2 * i);
                            int p2 = int(neighborReservoir.pathFlags.rcVertexLength() > 1 && neighborReservoir.pathFlags.pathLength() > 0);
                            numWorks += p2;
                            workMask |= p2 << (2 * i + 1);
                        }
                    }
                }
                else
                {
                    PathReservoir centralReservoir = reservoirs[centralOffset];

                    // fetch temporal reservoir
                    float2 motionVector = motionVectors[pixel];
                    int2 prevPixel = pixel;

                    if (gEnableTemporalReprojection)
                    {
                        prevPixel = pixel + motionVector * params.frameDim + (sampleNext2D(sg) * 1.f - 0.f);
                    }
                    int prevOffset = params.getReservoirOffset(prevPixel);

                    float3 prevV;
                    HitInfo temporalPrimaryHit = getPixelTemporalHitInfo(params, temporalVbuffer, prevPixel, prevOffset, prevV, pcf);
                    bool isValid = temporalPrimaryHit.isValid();

                    ShadingData temporalPrimarySd = getPixelTemporalShadingData(params, temporalVbuffer, prevPixel, temporalPrimaryHit, pcf);
                    isValid &= isValidGeometry(centralPrimarySd, temporalPrimarySd);


                    neighborValidMask[centralOffset].setValid(0, isValid);

                    if (ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid && isValid)
                    {
                        PathReservoir temporalReservoir = prevReservoirs[params.getReservoirOffset(prevPixel)];
                        int p1 = int(centralReservoir.pathFlags.rcVertexLength() > 1 && centralReservoir.pathFlags.pathLength() > 0);
                        numWorks += p1;
                        workMask |= p1;
                        int p2 = int(temporalReservoir.pathFlags.rcVertexLength() > 1 && temporalReservoir.pathFlags.pathLength() > 0);
                        numWorks += p2;
                        workMask |= p2 << 1;
                    }
                }
            }
        }

        const uint warpIdx = threadIdx >> 5;

        uint numWorksInWave = WaveActiveSum(numWorks);
        if (WaveIsFirstLane())
        {
            gWorkloadOffset[warpIdx] = numWorksInWave;
        }
        GroupMemoryBarrierWithGroupSync();

        if (threadIdx < kWarpCount)
        {
            uint numWorksInWave = gWorkloadOffset[threadIdx];
            gWorkloadOffset[threadIdx] = queue.addCounter(numWorksInWave);
        }
        GroupMemoryBarrierWithGroupSync();

        if (all(pixel < params.frameDim))
        {
            uint pathID = (pixel.y << 12) | pixel.x;
            uint dstIdx = gWorkloadOffset[warpIdx] + WavePrefixSum(numWorks);
            int j = 0;
            for (uint i = 0; i < 2 * neighborCount; i++)
            {
                if (workMask >> i & 1)
                {
                    queue.workload.Store((dstIdx + j) * 4, pathID + (i << 24));
                    j++;
                }
            }
        }
    }
}

cbuffer CB
{
    PathGenerator gPathGenerator;
}

[numthreads(256 /* kScreenTileDim.x * kScreenTileDim.y */, 1, 1)]
void main(
    uint3 groupID: SV_GroupID,
    uint3 groupThreadID: SV_GroupThreadID)
{
    gPathGenerator.execute(groupID.xy, groupThreadID.x);
}
