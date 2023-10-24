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


struct SuffixPathRetrace
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    ReSTIRPathTracerParams params; ///< Runtime parameters.
    // Resources
    ReSTIRPT restirpt;             ///< Settings for ReSTIR PT. Only valid if kReSTIRPT == true.


    Texture1D<float2> neighborOffsets;
    Texture2D<float2> motionVectors;

    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<PathReservoir> reservoirs;

    RWStructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    RWStructuredBuffer<uint> rcBufferOffsets;

    StructuredBuffer<PrefixGBuffer> prefixGBuffer;
    StructuredBuffer<PrefixGBuffer> prevPrefixGBuffer;

    RWStructuredBuffer<NeighborValidMask> neighborValidMask;

    int suffixReuseRoundId;
    int integrationPrefixId;

    RetraceWorkloadQueue queue;
    StructuredBuffer<uint> temporalNeighborPixels;

    // for linear growth of M when merging reservoirs


    int getNeighborCount(bool isTemporalReuse, bool finalGather)
    {
        return finalGather ? restirpt.subpathSettings.finalGatherSuffixCount 
        : (isTemporalReuse ? 1 : restirpt.subpathSettings.suffixSpatialNeighborCount);
    }

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int i)
    {
        if (i == -1) return pixel;

        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * restirpt.subpathSettings.suffixSpatialReuseRadius);

        return neighborPixel;
    }

    ShadingData getRcPrevShadingData(HitInfo rcPrevHit, float3 rcPrevWo, bool isPrevFrame)
    {
        ShadingData sd = {};
        if (rcPrevHit.isValid())
        {
            sd = loadShadingData(rcPrevHit, 0.f, -rcPrevWo, false, gPathTracer.createTextureSampler(), isPrevFrame);
        }
        return sd;
    }


    ShadingData getPixelPrefixLastSd(int2 pixel, bool isPrevFrame, bool useScratchSuffix, int offset, inout HitInfo prefixLastHit)
    
    {
        // temporal vertices?
        PrefixGBuffer sfPrevVertex;
        if (useScratchSuffix)
            sfPrevVertex = restirpt.prefixGBuffer[offset];
        else if (isPrevFrame)
            sfPrevVertex = prevPrefixGBuffer[offset];
        else
            sfPrevVertex = prefixGBuffer[offset];

        if (!sfPrevVertex.hit.isValid()) return {};

        prefixLastHit = sfPrevVertex.hit;
        return getRcPrevShadingData(prefixLastHit, sfPrevVertex.wo, isPrevFrame);
        
    }


#if RETRACE_SCHEDULE_TYPE == RETRACE_SCHEDULE_COMPACT
    void ReSTIR(const uint linearIndex)
    {
        // read number of paths
        uint totalPathCount = queue.counter.Load(0);
        if (linearIndex >= totalPathCount) return;
        uint pathID = queue.workload.Load(linearIndex * 4);
        int2 pixel = int2(pathID & 0xfff, pathID >> 12 & 0xfff);
        uint neighborIdx = pathID >> 24 & 0xf;
        int dstPrefixLength = int(pathID >> 28 & 0xf);
        // can reserve some bits for length change

        printSetPixel(pixel);
        logSetPixel(pixel);

        bool finalGather = suffixReuseRoundId == -1;

        SampleGenerator sg = SampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.suffixResamplingSgOffset() + (finalGather ? restirpt.finalGatherAdditionalOffset(integrationPrefixId) : suffixReuseRoundId));

        const uint centralOffset = params.getReservoirOffset(pixel);

        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;
        
        PathReservoir reservoir;

        int2 reprojPixel = pixel;
        bool temporalReuse = restirpt.subpathSettings.suffixTemporalReuse && suffixReuseRoundId == 0;
        if (temporalReuse)
        {
            float2 motionVector = motionVectors[pixel];
            reprojPixel = pixel + motionVector * params.frameDim + 0.5;
        }

        int neighborCount = getNeighborCount(temporalReuse, finalGather);

        int neighborId = neighborIdx / 2;

        int2 neighborPixel;
        if (finalGather)
            neighborPixel = unpackSearchedPixel(temporalNeighborPixels[neighborCount * centralOffset + neighborId]);
        else
            neighborPixel = getNextNeighborPixel(startIndex, reprojPixel, neighborId - (temporalReuse ? 1 : 0));

        uint neighborOffset = params.getReservoirOffset(neighborPixel);

        if (neighborIdx % 2 == 0)
        {
            if (finalGather)
                reservoir = restirpt.pathReservoirs[centralOffset];
            else if (temporalReuse)
                reservoir = reservoirs[centralOffset]; // if temporal reuse, buffers from previous pass are not swapped
            else
                reservoir = prevReservoirs[centralOffset]; // if temporal reuse, buffers from previous pass are not swapped
        }
        else
        {
            reservoir = prevReservoirs[neighborOffset];
        }

        int2 effectivePixel = neighborIdx % 2 == 0 ? neighborPixel : pixel;
        int effectiveOffset = neighborIdx % 2 == 0 ? neighborOffset : centralOffset;

        HitInfo PrefixLastHit;

        ShadingData PrefixLastSd = getPixelPrefixLastSd(effectivePixel, temporalReuse && neighborIdx == 0, finalGather && neighborIdx % 2 == 1,
                                                        effectiveOffset, PrefixLastHit);


        // modify flags
        ReSTIRPathFlags tempFlag = reservoir.pathFlags;
        if (restirpt.subpathSettings.adaptivePrefixLength)
        {
            int lengthChange = dstPrefixLength - tempFlag.prefixLength();
            tempFlag.insertPrefixLength(dstPrefixLength);
            if (tempFlag.rcVertexLength() != restirpt.kMaximumRcLength)
                tempFlag.insertRcVertexLength(tempFlag.rcVertexLength() + lengthChange);
            tempFlag.insertPathLength(tempFlag.pathLength() + lengthChange);
        }

        HitInfo rcPrevHit = HitInfo();
        float3 rcPrevWo = float3(0);
        float3 thp = 0.f;
        float dummy;

        if (tempFlag.pathLength() <= kMaxSurfaceBounces)
        {
            thp = TraceReplayedPath(PrefixLastHit,

                                    PrefixLastSd,

                                    tempFlag,
                                    reservoir.initRandomSeed, reservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, temporalReuse && (neighborIdx == 0), dummy, true);
        }
        rcBufferOffsets[2 * neighborCount * centralOffset + neighborIdx] = linearIndex;
        reconnectionDataBuffer[linearIndex] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
    }
#else
    void ReSTIR(const uint2 pixel)
    {
        bool temporalReuse = restirpt.subpathSettings.suffixTemporalReuse && suffixReuseRoundId == 0;
        bool finalGather = suffixReuseRoundId == -1;
        SampleGenerator sg = SampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.suffixResamplingSgOffset() + (finalGather ? restirpt.finalGatherAdditionalOffset(integrationPrefixId) : suffixReuseRoundId));

        const uint centralOffset = params.getReservoirOffset(pixel);

        PathReservoir centralReservoir;
        if (finalGather)
            centralReservoir = restirpt.pathReservoirs[centralOffset];
        else if (temporalReuse)
            centralReservoir = reservoirs[centralOffset]; // if temporal reuse, buffers from previous pass are not swapped
        else
            centralReservoir = prevReservoirs[centralOffset];

        HitInfo centralPrefixLastHit;

        ShadingData centralPrefixLastSd = getPixelPrefixLastSd(pixel, false, finalGather, centralOffset, centralPrefixLastHit);
        if (all(centralPrefixLastSd.V == 0.f)) return;

        int neighborCount = getNeighborCount(temporalReuse, finalGather);

        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

        int2 reprojPixel = pixel;
        if (temporalReuse)
        {
            float2 motionVector = motionVectors[pixel];
            reprojPixel = pixel + motionVector * params.frameDim + 0.5f;        
        }

        bool currentUserFlagSet = centralReservoir.pathFlags.isUserFlagSet();

        for (int i = 0; i < neighborCount; ++i)
        {
            int2 neighborPixel;
            if (finalGather)
                neighborPixel = unpackSearchedPixel(temporalNeighborPixels[neighborCount * centralOffset + i]);
            else
                neighborPixel = getNextNeighborPixel(startIndex, reprojPixel, i - (temporalReuse ? 1 : 0));

            if (!isValidScreenRegion(params, neighborPixel))
            {
                neighborValidMask[centralOffset].setValid(i, false);
                continue;
            }

            const uint neighborOffset = params.getReservoirOffset(neighborPixel);
            PathReservoir neighborReservoir = prevReservoirs[neighborOffset];

            //////////////////////////////////////

            HitInfo neighborPrefixLastHit;

            ShadingData neighborPrefixLastSd = getPixelPrefixLastSd(neighborPixel, temporalReuse, false, neighborOffset, neighborPrefixLastHit);
            if (all(neighborPrefixLastSd.V == 0.f))
            {
                neighborValidMask[centralOffset].setValid(i, false);
                continue;
            }

            neighborValidMask[centralOffset].setValid(i, true);

            if (ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
            {
                if (centralReservoir.pathFlags.rcVertexLength() > 1 + centralReservoir.pathFlags.prefixLength() &&
                    centralReservoir.pathFlags.pathLength() >= centralReservoir.pathFlags.prefixLength() &&
                    neighborReservoir.pathFlags.pathTreeLength() >= restirpt.minimumPrefixLength)
                {
                    HitInfo rcPrevHit = HitInfo();
                    float3 rcPrevWo = float3(0.f);
                    float3 thp = 0.f;
                    float dummy;

                    ReSTIRPathFlags tempFlag = centralReservoir.pathFlags;
                    // modify the path length
                    if (restirpt.subpathSettings.adaptivePrefixLength)
                    {
                        int dstPrefixLength = neighborReservoir.pathFlags.prefixLength();
                        int lengthChange = dstPrefixLength - centralReservoir.pathFlags.prefixLength();
                        if (neighborReservoir.pathFlags.pathLength() + lengthChange > kMaxSurfaceBounces)
                            tempFlag.insertPathLength(0); // nullify the sample
                        tempFlag.insertPrefixLength(dstPrefixLength);
                        if (tempFlag.rcVertexLength() != restirpt.kMaximumRcLength)
                            tempFlag.insertRcVertexLength(tempFlag.rcVertexLength() + lengthChange);
                        tempFlag.insertPathLength(tempFlag.pathLength() + lengthChange);
                    }

                    if (tempFlag.pathLength() <= kMaxSurfaceBounces)
                    {
                        thp = TraceReplayedPath(neighborPrefixLastHit,

                                                neighborPrefixLastSd, 

                                                tempFlag,
                                                centralReservoir.initRandomSeed, centralReservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, temporalReuse, dummy, true);
                    }
                    rcBufferOffsets[2 * neighborCount * centralOffset + 2 * i] = 2 * neighborCount * centralOffset + 2 * i;
                    reconnectionDataBuffer[2 * neighborCount * centralOffset + 2 * i] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
                }

                if (neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength() &&
                    neighborReservoir.pathFlags.pathLength() >= neighborReservoir.pathFlags.prefixLength() &&                
                    centralReservoir.pathFlags.pathTreeLength() >= restirpt.minimumPrefixLength)
                {
                    HitInfo rcPrevHit = HitInfo();
                    float3 rcPrevWo = float3(0.f);
                    float3 thp = 0.f;
                    float dummy;
                    ReSTIRPathFlags tempFlag = neighborReservoir.pathFlags;
                    if (restirpt.subpathSettings.adaptivePrefixLength)
                    {
                        int dstPrefixLength = centralReservoir.pathFlags.prefixLength();
                        int lengthChange = dstPrefixLength - neighborReservoir.pathFlags.prefixLength();
                        if (centralReservoir.pathFlags.pathLength() + lengthChange > kMaxSurfaceBounces)
                            tempFlag.insertPathLength(0); // nullify the sample
                        tempFlag.insertPrefixLength(dstPrefixLength);
                        if (tempFlag.rcVertexLength() != restirpt.kMaximumRcLength)
                            tempFlag.insertRcVertexLength(tempFlag.rcVertexLength() + lengthChange);
                        tempFlag.insertPathLength(tempFlag.pathLength() + lengthChange);
                    }

                    if (tempFlag.pathLength() <= kMaxSurfaceBounces)
                    {
                        thp = TraceReplayedPath(centralPrefixLastHit,

                                                centralPrefixLastSd, 

                                                tempFlag,
                                                neighborReservoir.initRandomSeed, neighborReservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, false, dummy, true);
                    }

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
    SuffixPathRetrace gSuffixPathRetrace;
}

#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
[numthreads(256, 1, 1)]
#else
[numthreads(16, 16, 1)]
#endif
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
    gSuffixPathRetrace.execute(dispatchThreadId.x);
#else
    gSuffixPathRetrace.execute(dispatchThreadId.xy);
#endif
}
