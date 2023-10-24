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
import ResamplingCommon;
struct SpatialResampling
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    ReSTIRPathTracerParams params;
    // Resources
    ReSTIRPT restirpt; ///< Settings for ReSTIR PT. Only valid if kReSTIRPT == true.

    
    Texture2D<PackedHitInfo> vbuffer;                     ///< Fullscreen V-buffer for the primary hits.

    Texture1D<float2> neighborOffsets;

    RWStructuredBuffer<PathReservoir> reservoirs;
    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<PathReservoir> suffixReservoirs;

    StructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<uint> rcBufferOffsets;
    StructuredBuffer<NeighborValidMask> neighborValidMask;

    StructuredBuffer<ReSTIRPTVertex> prevPathVertices;
    RWStructuredBuffer<ReSTIRPTVertex> pathVertices;

    int gSpatialRoundId;
    int  gNumSpatialRounds;
    // for default pattern
    int gNeighborCount;
    float gGatherRadius;
    bool gAdditionalSpatial;

    int getNeighborCount()
    {
        return gNeighborCount; // does not include self
    }

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int i)
    {
        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * gGatherRadius);
 
        return neighborPixel;
    }

    void TestNullShift(const uint2 pixel, PathReservoir r, ShadingData sd)
    {
        ReconnectionData rcData;
        rcData.initPseudoData();

        if (ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid && 
        r.pathFlags.rcVertexLength() > 1)
        {
            const uint centralOffset = params.getReservoirOffset(pixel);
            rcData = reconnectionDataBuffer[rcBufferOffsets[2 * getNeighborCount() * centralOffset]];
        }

        float shiftJacobian;
        float3 shiftedIntegrand = computeShiftedIntegrand(restirpt, params, shiftJacobian, r, rcData, 0);
        float3 originalIntegrand = r.integrand;

        print("\n", 0);
        print("Original Integrand", originalIntegrand);
        print("Shift Jacobian", shiftJacobian);
        print("Shifted Integrand", shiftedIntegrand);
        print("\n", 0);

        if (!r.pathFlags.decodeIsDeltaEvent(true) && !r.pathFlags.decodeIsDeltaEvent(false))
        {
            assert(all(shiftedIntegrand == 0) && all(originalIntegrand == 0) || length(shiftedIntegrand - originalIntegrand) < 0.001);
            assert(shiftJacobian == 0.f || shiftJacobian > 0.99);
        }
    }

    void ReSTIR(const uint2 pixel)
    {
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.spatialSgOffset(gSpatialRoundId, gAdditionalSpatial));

        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = prevReservoirs[centralOffset];

        float3 centralPrimaryWo;
        HitInfo centralPrimaryHit = getPixelHitInfo(params, vbuffer, pixel, centralOffset, centralPrimaryWo);

        if (!centralPrimaryHit.isValid()) return;

        PathReservoir dstReservoir;
        dstReservoir.init();

        int neighborCount = getNeighborCount();
        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;
        
        int validNeighborCount = 0;
        float canonicalWeight = 1;

        uint selectedNeighborOffset = uint(-1);

        for (int i = 0; i < neighborCount; ++i)
        {
            if (!neighborValidMask[centralOffset].isValid(i)) continue;
            int2 neighborPixel = getNextNeighborPixel(startIndex, pixel, i);
            const uint neighborOffset = params.getReservoirOffset(neighborPixel);

            float3 neighborPrimaryWo;
            HitInfo neighborPrimaryHit = getPixelHitInfo(params, vbuffer, neighborPixel, neighborOffset, neighborPrimaryWo);

            PathReservoir neighborReservoir = prevReservoirs[neighborOffset];

            validNeighborCount++;

            ReconnectionData rcData;
            if (centralReservoir.pathFlags.rcVertexLength() > 1 && ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
            {

                rcData = reconnectionDataBuffer[rcBufferOffsets[2 * neighborCount * centralOffset + 2 * i]];
            }
            else
            {
                rcData.pathThroughput = 1.f;
                rcData.rcPrevHit = neighborPrimaryHit;
                rcData.rcPrevWo = neighborPrimaryWo; 
            }

            float canonicalAtOtherJacobian = 1.f;
            float3 canonicalAtOtherIntegrand = computeShiftedIntegrand(restirpt, params, canonicalAtOtherJacobian, centralReservoir, rcData, 0);
            float canonicalAtOtherPHat = PathRisState::toScalar(canonicalAtOtherIntegrand) * canonicalAtOtherJacobian;

            canonicalWeight += 1;
            canonicalWeight -= (canonicalAtOtherPHat > 0.f) 
                ? canonicalAtOtherPHat * neighborReservoir.M /
                    (canonicalAtOtherPHat * neighborReservoir.M + centralReservoir.M * PathRisState::toScalar(centralReservoir.integrand) / (neighborCount))
                : 0;

            if (neighborReservoir.pathFlags.rcVertexLength() > 1 && ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
            {
                rcData = reconnectionDataBuffer[rcBufferOffsets[2 * neighborCount * centralOffset + 2 * i + 1]];
            }
            else
            {
                rcData.pathThroughput = 1.f;
                rcData.rcPrevHit = centralPrimaryHit;
                rcData.rcPrevWo = centralPrimaryWo;
            }

            bool selected = shiftAndMergeReservoir(restirpt, params, dstReservoir, neighborReservoir, rcData, sg, (float)centralReservoir.M / neighborCount);
            selectedNeighborOffset = selected ? neighborOffset : selectedNeighborOffset;
        }

        // merge central reservoir itself
        bool selected = dstReservoir.merge(centralReservoir.integrand, 1.f, centralReservoir, sg, canonicalWeight);
        selectedNeighborOffset = selected ? centralOffset : selectedNeighborOffset;

        // "Finalize" the weight (convert from w-sum to unbiased contribution weight W)
        dstReservoir.finalizeGRIS();
        dstReservoir.weight /= (validNeighborCount + 1); // compensate for the fact that pairwise resampling MIS was not divided by (k+1)
        dstReservoir.weight = (isnan(dstReservoir.weight) || isinf(dstReservoir.weight))
            ? 0.f
            : max(0.f, dstReservoir.weight);

        reservoirs[centralOffset] = dstReservoir;
    }

    /** Entry point for merge pass.
        \param[in] pixel Pixel coordinates.
    */
    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        ReSTIR(pixel);
    }
};

cbuffer CB
{
    SpatialResampling gSpatialResampling;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gSpatialResampling.execute(dispatchThreadId.xy);
}
