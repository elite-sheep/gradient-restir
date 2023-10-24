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
import LoadShadingData;
import HybridShiftTraceRays;
import RetraceWorkloadQueue;
import ResamplingCommon;
#include "RetraceScheduleDefinition.slangh"


/** TODO.
*/
struct TemporalPathRetrace
{
    // Resources
    ReSTIRPathTracerParams params; ///< Runtime parameters.
    ReSTIRPT restirpt;             ///< Settings for ReSTIR PT. Only valid if kReSTIRPT == true.

    Texture2D<PackedHitInfo> vbuffer; ///< Fullscreen V-buffer for the primary hits.
    Texture2D<PackedHitInfo> temporalVbuffer;


    Texture2D<float2> motionVectors;

    StructuredBuffer<PathReservoir> reservoirs;
    StructuredBuffer<PathReservoir> prevReservoirs;
    RWStructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<uint2> keyValueBuffer;
    RWStructuredBuffer<NeighborValidMask> neighborValidMask;
    RWStructuredBuffer<uint> rcBufferOffsets;

    float3 prevCameraU;
    float3 prevCameraV;
    float3 prevCameraW; ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float prevJitterX;  ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
    float prevJitterY;  ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).

    int gNumSpatialRounds;
    bool gEnableTemporalReprojection;
    bool gNoResamplingForTemporalReuse;
    uint gTemporalHistoryLength;

    RetraceWorkloadQueue queue;

#if RETRACE_SCHEDULE_TYPE == RETRACE_SCHEDULE_COMPACT
    void ReSTIR(const uint linearIndex)
    {
        // read number of paths
        uint totalPathCount = queue.counter.Load(0);
        if (linearIndex >= totalPathCount) return;
        uint pathID = queue.workload.Load(linearIndex * 4);
        int2 pixel = int2(pathID & 0xfff, pathID >> 12 & 0xfff);
        uint neighborId = pathID >> 24 & 0xff;
        const PrevCameraFrame pcf = { prevCameraU, prevCameraV, prevCameraW, prevJitterX, prevJitterY };

        printSetPixel(pixel);
        logSetPixel(pixel);

        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.temporalSgOffset());

        const uint centralOffset = params.getReservoirOffset(pixel);

        float2 motionVector = motionVectors[pixel];
        int2 prevPixel = pixel;
        if (gEnableTemporalReprojection)
        {
            prevPixel = pixel + motionVector * params.frameDim + (sampleNext2D(sg) * 1.f - 0.f);
        }
        const uint temporalOffset = params.getReservoirOffset(prevPixel);

        PathReservoir reservoir;
        if (neighborId % 2)
        {
            reservoir = prevReservoirs[temporalOffset];
        }
        else
        {
            reservoir = reservoirs[centralOffset];
        }

        HitInfo primaryHit;

        ShadingData primarySd = getPixelShadingDataGeneral(params, vbuffer, temporalVbuffer, neighborId % 2 ? pixel : prevPixel, primaryHit, neighborId % 2 == 0, pcf);

        HitInfo rcPrevHit = HitInfo();
        float3 rcPrevWo = float3(0);
        float dummy;
        float3 thp = TraceReplayedPath(primaryHit,

                                       primarySd, 

                                       reservoir.pathFlags,
                                       reservoir.initRandomSeed, reservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, neighborId % 2 == 0, dummy);

        rcBufferOffsets[2 * centralOffset + neighborId] = linearIndex;
        reconnectionDataBuffer[linearIndex] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
    }
#else
    bool ReSTIR(const uint2 pixel)
    {
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.temporalSgOffset());
        const PrevCameraFrame pcf = { prevCameraU, prevCameraV, prevCameraW, prevJitterX, prevJitterY };

        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = reservoirs[centralOffset];

        HitInfo centralPrimaryHit;
        ShadingData centralPrimarySd = getPixelShadingData(params, vbuffer, pixel, centralPrimaryHit);

        if (!centralPrimaryHit.isValid()) return false;

        float2 motionVector = motionVectors[pixel];
        int2 prevPixel = pixel;
        if (gEnableTemporalReprojection)
        {
            prevPixel = pixel + motionVector * params.frameDim + (sampleNext2D(sg) * 1.f - 0.f);
        }
        int prevOffset = params.getReservoirOffset(prevPixel);

        if (!isValidScreenRegion(params, prevPixel)) return false;

        HitInfo temporalPrimaryHit;
        ShadingData temporalPrimarySd = getPixelTemporalShadingData(params, temporalVbuffer, prevPixel, temporalPrimaryHit, pcf);
        if (!temporalPrimaryHit.isValid() || !isValidGeometry(centralPrimarySd, temporalPrimarySd)) return false;


        if (ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
        {
            PathReservoir temporalReservoir = prevReservoirs[prevOffset];

            // write to memory
            if (centralReservoir.pathFlags.rcVertexLength() > 1 && centralReservoir.pathFlags.pathLength() > 0)
            {
                HitInfo rcPrevHit = HitInfo();
                float3 rcPrevWo = float3(0.f);
                float3 thp = 1.f;
                float dummy;
                thp = TraceReplayedPath(temporalPrimaryHit,

                                        temporalPrimarySd, 

                                        centralReservoir.pathFlags,
                                        centralReservoir.initRandomSeed, centralReservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, true, dummy);

                rcBufferOffsets[2 * centralOffset] = 2 * centralOffset;
                reconnectionDataBuffer[2 * centralOffset] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
            }

            if (temporalReservoir.pathFlags.rcVertexLength() > 1 && temporalReservoir.pathFlags.pathLength() > 0)
            {
                HitInfo rcPrevHit = HitInfo();
                float3 rcPrevWo = float3(0.f);
                float3 thp = 1.f;
                float dummy;
                thp = TraceReplayedPath(centralPrimaryHit,

                                        centralPrimarySd,

                                        temporalReservoir.pathFlags,
                                        temporalReservoir.initRandomSeed, temporalReservoir.suffixInitRandomSeed, rcPrevHit, rcPrevWo, false, dummy);
                rcBufferOffsets[2 * centralOffset + 1] = 2 * centralOffset + 1;
                reconnectionDataBuffer[2 * centralOffset + 1] = ReconnectionData(rcPrevHit, rcPrevWo, thp);
            }
        }
        return true;
    }
#endif

/** Entry point for merge pass.
    \param[in] pixel Pixel coordinates.
*/
#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
    void execute(const uint index)
    {
        if (index >= 2 * params.frameDim.x * params.frameDim.y) return;
        ReSTIR(index);
    }
#else
    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        bool isValid = ReSTIR(pixel);
        const uint centralOffset = params.getReservoirOffset(pixel);
        neighborValidMask[centralOffset].setValid(0, isValid);
    }
#endif
};

cbuffer CB
{
    TemporalPathRetrace gTemporalPathRetrace;
}

#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
[numthreads(256, 1, 1)]
#else
[numthreads(16, 16, 1)]
#endif
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
#if RETRACE_SCHEDULE_TYPE != RETRACE_SCHEDULE_NAIVE
    gTemporalPathRetrace.execute(dispatchThreadId.x);
#else
    gTemporalPathRetrace.execute(dispatchThreadId.xy);
#endif
}
