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
import Shift;
import Scene.HitInfo;
import PathReservoir;
import PathTracer;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import LoadShadingData;
import HybridShiftTraceRays;
import RetraceWorkloadQueue;
import ResamplingCommon;
#include "RetraceScheduleDefinition.slangh"


struct SpatialPathRetrace
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    ReSTIRPathTracerParams params; ///< Runtime parameters.
    // Resources
    ReSTIRPT restirpt;             ///< Settings for ReSTIR PT. Only valid if kReSTIRPT == true.


    Texture2D<PackedHitInfo> vbuffer; ///< Fullscreen V-buffer for the primary hits.


    Texture1D<float2> neighborOffsets;

    StructuredBuffer<PathReservoir> prevReservoirs;
    RWStructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<uint2> keyValueBuffer;
    RWStructuredBuffer<NeighborValidMask> neighborValidMask;
    RWStructuredBuffer<uint> rcBufferOffsets;

    int gSpatialRoundId;
    int  gNumSpatialRounds;
    // for default pattern
    int gNeighborCount;
    float gGatherRadius;
    bool gAdditionalSpatial;

    RetraceWorkloadQueue queue;

    int getNeighborCount()
    {
        return gNeighborCount;
    }

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int i)
    {
        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * gGatherRadius);

        return neighborPixel;
    }

#if RETRACE_SCHEDULE_TYPE == RETRACE_SCHEDULE_COMPACT
    void ReSTIR(const uint linearIndex)
    {
        // read number of paths
        uint totalPathCount = queue.counter.Load(0);
        if (linearIndex >= totalPathCount) return;
        uint pathID = queue.workload.Load(linearIndex * 4);
        int2 pixel = int2(pathID & 0xfff, pathID >> 12 & 0xfff);
        uint neighborId = pathID >> 24 & 0xff;

        printSetPixel(pixel);
        logSetPixel(pixel);

        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.spatialSgOffset(gSpatialRoundId, gAdditionalSpatial));

        const uint centralOffset = params.getReservoirOffset(pixel);

        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;
        int2 neighborPixel = getNextNeighborPixel(startIndex, pixel, neighborId / 2);
        const uint neighborOffset = params.getReservoirOffset(neighborPixel);

        PathReservoir reservoir = prevReservoirs[neighborId%2 ? neighborOffset : centralOffset];

        HitInfo primaryHit;

        ShadingData primarySd = getPixelShadingData(params, vbuffer, neighborId % 2 ? pixel : neighborPixel, primaryHit);

        if (neighborId % 2)
        {
            reservoir.pathFlags.insertPathTreeLength(1);
        }
        else
        {
            reservoir.pathFlags.insertPathTreeLength(0);
        }

        HitInfo rcPrevHit = HitInfo();
        float3 rcPrevWo = float3(0);
        float dummy;

        float3 thp = TraceReplayedPath(primaryHit,

                                       primarySd, 

                                       reservoir.pathFlags,
                                       reservoir.initRandomSeed, reservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, false, dummy);

        rcBufferOffsets[2 * getNeighborCount() * centralOffset + neighborId] = linearIndex;
        reconnectionDataBuffer[linearIndex] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
    }
#else
    void ReSTIR(const uint2 pixel)
    {
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.spatialSgOffset(gSpatialRoundId, gAdditionalSpatial));

        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = prevReservoirs[centralOffset];

        HitInfo centralPrimaryHit;

        ShadingData centralPrimarySd = getPixelShadingData(params, vbuffer, pixel, centralPrimaryHit);
        
        if (!centralPrimaryHit.isValid()) return;

        int neighborCount = getNeighborCount();

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

            //print("originalL", neighborReservoir.integrand);


            ShadingData neighborPrimarySd = getPixelShadingData(params, vbuffer, neighborPixel, neighborPrimaryHit);
            if (!neighborPrimaryHit.isValid() || !isValidGeometry(centralPrimarySd, neighborPrimarySd))

            {
                neighborValidMask[centralOffset].setValid(i, false);
                continue;
            }

            neighborValidMask[centralOffset].setValid(i, true);

            if (ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
            {
                if (centralReservoir.pathFlags.rcVertexLength() > 1 && centralReservoir.pathFlags.pathLength() > 0)
                {
                    HitInfo rcPrevHit = HitInfo();
                    float3 rcPrevWo = float3(0.f);
                    float3 thp = 1.f;
                    float dummy;
                    thp = TraceReplayedPath(neighborPrimaryHit, 
                                            neighborPrimarySd, 
                                            centralReservoir.pathFlags,
                                            centralReservoir.initRandomSeed, centralReservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, false, dummy);

                    rcBufferOffsets[2 * neighborCount * centralOffset + 2 * i] = 2 * neighborCount * centralOffset + 2 * i;
                    reconnectionDataBuffer[2 * neighborCount * centralOffset + 2 * i] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
                }

                if (neighborReservoir.pathFlags.rcVertexLength() > 1 && neighborReservoir.pathFlags.pathLength() > 0)
                {
                    HitInfo rcPrevHit = HitInfo();
                    float3 rcPrevWo = float3(0.f);
                    float3 thp = 1.f;
                    float dummy;
                    thp = TraceReplayedPath(centralPrimaryHit,

                                            centralPrimarySd,

                                            neighborReservoir.pathFlags,
                                            neighborReservoir.initRandomSeed, neighborReservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, false, dummy);

                    rcBufferOffsets[2 * neighborCount * centralOffset + 2 * i + 1] = 2 * neighborCount * centralOffset + 2 * i + 1;
                    reconnectionDataBuffer[2 * neighborCount * centralOffset + 2 * i + 1] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
                }
            }
        }
    }
#endif
    /** Entry point for merge pass.
        \param[in] pixel Pixel coordinates.
    */
#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
    void execute(const uint index)
    {
        ReSTIR(index);
    }
#else
    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        ReSTIR(pixel);
    }
#endif
};

cbuffer CB
{
    SpatialPathRetrace gSpatialPathRetrace;
}

#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
[numthreads(256, 1, 1)]
#else
[numthreads(16, 16, 1)]
#endif
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
    gSpatialPathRetrace.execute(dispatchThreadId.x);
#else
    gSpatialPathRetrace.execute(dispatchThreadId.xy);
#endif
}
