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
import ResamplingCommon;


/** TODO.
 */
struct TemporalResampling
{
    // Resources
    ReSTIRPathTracerParams params;
    ReSTIRPT restirpt; ///< Settings for ReSTIR PT. Only valid if kReSTIRPT == true.

    Texture2D<PackedHitInfo> vbuffer;                     ///< Fullscreen V-buffer for the primary hits.
    Texture2D<PackedHitInfo> temporalVbuffer;
    
    Texture2D<float2> motionVectors;

    RWStructuredBuffer<PathReservoir> reservoirs;
    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<uint> rcBufferOffsets;

    StructuredBuffer<NeighborValidMask> neighborValidMask;
    StructuredBuffer<ReSTIRPTVertex> prevPathVertices;
    RWStructuredBuffer<ReSTIRPTVertex> pathVertices;

    float3 prevCameraU;
    float3 prevCameraV;
    float3 prevCameraW; ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float prevJitterX; ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
    float prevJitterY; ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).

    int  gNumSpatialRounds;
    bool gEnableTemporalReprojection;

    uint gTemporalHistoryLength;
    bool gNoResamplingForTemporalReuse;

    void TestNullShift(const uint2 pixel, PathReservoir r, ShadingData sd)
    {
        ReconnectionData rcData;
        rcData.initPseudoData();

        if (ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid &&
            r.pathFlags.rcVertexLength() > 1)
        {
            const uint centralOffset = params.getReservoirOffset(pixel);
            rcData = reconnectionDataBuffer[rcBufferOffsets[2 * centralOffset]];
        }

        float shiftJacobian;
        float3 shiftedIntegrand = computeShiftedIntegrand(restirpt, params, shiftJacobian, r, rcData, 0, false, true);
        float3 originalIntegrand = r.integrand;

        print("\n", 0);
        print("Original Integrand", originalIntegrand);
        print("Shift Jacobian", shiftJacobian);
        print("Shifted Integrand", shiftedIntegrand);
        print("\n", 0);
    }
    
    void ReSTIR(const uint2 pixel)
    {
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.temporalSgOffset());

        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = reservoirs[centralOffset];
        centralReservoir.M = 1; //reset M in case of suffix reuse
        
        PathReservoir dstReservoir;
        dstReservoir.init();

        float3 centralPrimaryWo;
        HitInfo centralPrimaryHit = getPixelHitInfo(params, vbuffer, pixel, centralOffset, centralPrimaryWo);
        if (!centralPrimaryHit.isValid()) return;
        
        // do a null temporal update for the current reservoir
        // TestNullShift(pixel, centralReservoir, centralPrimarySd);

        if (!neighborValidMask[centralOffset].isValid(0)) return;

        // fetch temporal reservoir
        float2 motionVector = motionVectors[pixel];
        int2 prevPixel = pixel;

        if (gEnableTemporalReprojection)
        {
            prevPixel = pixel + motionVector * params.frameDim + (sampleNext2D(sg) * 1.f - 0.f);
        }

        const PrevCameraFrame pcf = { prevCameraU, prevCameraV, prevCameraW, prevJitterX, prevJitterY}; 

        const uint prevOffset = params.getReservoirOffset(prevPixel);

        float3 temporalPrimaryWo;
        HitInfo temporalPrimaryHit = getPixelTemporalHitInfo(params, temporalVbuffer, prevPixel, prevOffset, temporalPrimaryWo, pcf);

        static const bool doTemporalUpdate = restirpt.kTemporalUpdateForDynamicScene;
        static const bool doInverseTemporalUpdate = restirpt.kUsePrevFrameSceneData ? restirpt.kTemporalUpdateForDynamicScene : false;
        // if we don't have prev frame scene data, then inverse temporal update is meaningless

        PathReservoir temporalReservoir = prevReservoirs[prevOffset];
        temporalReservoir.M = min(gTemporalHistoryLength * centralReservoir.M, temporalReservoir.M);

        bool selectedPrev = false;

        if (!gNoResamplingForTemporalReuse)
        {
            ReconnectionData rcData;
            if (centralReservoir.pathFlags.rcVertexLength() > 1 && ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
            {
                rcData = reconnectionDataBuffer[rcBufferOffsets[2*centralOffset]];
            }
            else
            {
                rcData.pathThroughput = 1.f;
                rcData.rcPrevHit = temporalPrimaryHit;
                rcData.rcPrevWo = temporalPrimaryWo;
            }

            float canonicalAtOtherJacobian = 1.f;
            float3 canonicalAtOtherIntegrand = computeShiftedIntegrand(restirpt, params, canonicalAtOtherJacobian, centralReservoir, rcData, 0, true, doInverseTemporalUpdate);
            float canonicalAtOtherPHat = PathRisState::toScalar(canonicalAtOtherIntegrand) * canonicalAtOtherJacobian;

            float canonicalWeight = (canonicalAtOtherPHat > 0.f)
                ? 1.f - (canonicalAtOtherPHat * temporalReservoir.M /
                    (canonicalAtOtherPHat * temporalReservoir.M + centralReservoir.M * PathRisState::toScalar(centralReservoir.integrand)))
                : 1.f;

            if (temporalReservoir.pathFlags.rcVertexLength() > 1 && ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
            {

                rcData = reconnectionDataBuffer[rcBufferOffsets[2 * centralOffset+1]];
            }
            else
            {
                rcData.pathThroughput = 1.f;
                rcData.rcPrevHit = centralPrimaryHit;
                rcData.rcPrevWo = centralPrimaryWo;
            }

            selectedPrev = shiftAndMergeReservoir(restirpt, params, dstReservoir,
                                                  temporalReservoir, rcData, sg, centralReservoir.M, doTemporalUpdate);

            selectedPrev = !dstReservoir.merge(centralReservoir.integrand, 1.f, centralReservoir, sg, canonicalWeight);
        }
        else
        {
            selectedPrev = !dstReservoir.merge(centralReservoir.integrand, 1.f, centralReservoir, sg, float(centralReservoir.M) / (temporalReservoir.M + centralReservoir.M));
            selectedPrev = dstReservoir.merge(temporalReservoir.integrand, 1.f, temporalReservoir, sg, float(temporalReservoir.M) / (temporalReservoir.M + centralReservoir.M));
        }

        dstReservoir.finalizeGRIS();
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
    TemporalResampling gTemporalResampling;
}


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gTemporalResampling.execute(dispatchThreadId.xy);
}
