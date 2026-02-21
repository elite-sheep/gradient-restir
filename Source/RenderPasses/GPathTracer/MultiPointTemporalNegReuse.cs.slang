// Copyright @yucwang 2025

import Scene.RaytracingInline;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;
import Scene.HitInfo;
import Utils.Math.Ray;

import GPathTracer;
import Params;
import PathShift;
import PathState;
import RISDataTypes;
import StaticParams;
import Utils;

import MultiPointReSTIR;

MultiPointTemporalReuse gGradientIntegrator;
ParameterBlock<GPathTracer> gPathTracer;
ParameterBlock<MultiPointReSTIR> multiPointReSTIR;
uniform uint shiftDir;
uniform uint pathsPerPixel;
Texture2D<uint2> neighbourPixels;

struct RTQueryImpl : GPathTracer::IRTQuery
{
    bool traceRay(const Ray ray, out HitInfo hit, out float hitT)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff, false);
    }

    bool traceVisibilityRay(const Ray ray)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff, false);
    }
};

struct MultiPointTemporalReuse
{
    void run(uint2 pixel)
    {
        uint2 shift = (shiftDir == 0) ? uint2(1, 0) : uint2(0, 1);
        uint2 frameDim = gPathTracer.params.frameDim;
        SampleGenerator risSg = SampleGenerator(pixel, gPathTracer.params.seed * (kNumInitialSamples + 2));
        uint basePathBufferIndex = getBufferOffset(pixel, frameDim, kNumInitialSamples);
        uint2 shiftPixel = pixel + shift;
        uint shiftPathBufferIndex = getBufferOffset(shiftPixel, frameDim, kNumInitialSamples);
        uint baseBufferIndex = min(basePathBufferIndex, shiftPathBufferIndex);

        uint basePixelBufferIndex = getBufferOffset(pixel, frameDim);
        uint shiftPixelBufferIndex = getBufferOffset(shiftPixel, frameDim);

        uint selectedPathBufferIndex = getBufferOffset(pixel, frameDim, pathsPerPixel);
        // bool haveSpatialNeighbour = !all(neighbourPixels[pixel] == pixel) && kNumSpatialNeighbours > 0;
        // uint finalSelectedPathBufferIndex = haveSpatialNeighbour ? getBufferOffset(pixel, frameDim, pathsPerPixel) :
        //                                getBufferOffset(pixel, frameDim, pathsPerPixel) + pathsPerPixel - 1;
        uint finalSelectedPathBufferIndex = selectedPathBufferIndex;

        if (isInFrame(pixel, frameDim) && isInFrame(shiftPixel, frameDim))
        {
            RTQueryImpl rtq;
            PathReservoir negReservoir = multiPointReSTIR.negReservoirs[selectedPathBufferIndex];
            PathSampleRecord negSelectedPath = multiPointReSTIR.negPaths[selectedPathBufferIndex];
            PathReservoir negTemporalReservoir = multiPointReSTIR.negSampledTemporalReservoirs[basePixelBufferIndex];
            bool negFoundNeighbour = !negTemporalReservoir.isNullSample;
            if (kUseTemporalReuse && negFoundNeighbour)
            {
                float2 motionVector = (multiPointReSTIR.gPathTracer.motionVectors[pixel]
                                        + multiPointReSTIR.gPathTracer.motionVectors[shiftPixel]) * 0.5f;
            #if SHIFT_MAPPING_TYPE == 2
                if (multiPointReSTIR.primaryDepth[pixel] < multiPointReSTIR.primaryDepth[shiftPixel])
                {
                    motionVector = multiPointReSTIR.gPathTracer.motionVectors[pixel];
                }
                else
                {
                    motionVector = multiPointReSTIR.gPathTracer.motionVectors[shiftPixel];
                }
                float2 screenMotion = motionVector * float2(frameDim);
                float2 prevFracCenter = float2(pixel) + float2(0.5f) + screenMotion;
                float2 prevFracPixel = prevFracCenter - float2(0.5f);
            #else
                int2 prevBasePixel = pixel;
                prevBasePixel = int2(pixel + motionVector * frameDim + sampleNext2D(risSg));
                uint prevBufferIndex = getBufferOffset(prevBasePixel, frameDim);
                int2 prevShiftPixel = prevBasePixel + shift;

                int2 posPrevBasePixel = prevBasePixel;
                int2 posPrevShiftPixel = prevShiftPixel;
                uint posPrevBufferIndex = prevBufferIndex;
                uint posPrevShiftBufferIndex = getBufferOffset(posPrevShiftPixel, frameDim);

                int2 negPrevBasePixel = prevBasePixel;
                int2 negPrevShiftPixel = prevShiftPixel;
                uint negPrevBufferIndex = prevBufferIndex;
                uint negPrevShiftBufferIndex = getBufferOffset(negPrevShiftPixel, frameDim);
            #endif

                if (isInFrame(shiftPixel, frameDim) && negFoundNeighbour)
                {
                    RTQueryImpl rtq;
                    PathReservoir negTmpReservoir;
                    negTmpReservoir.init();
                    for (int i = 0; i < 2; ++i)
                    {
                        PathReservoir curReservoir = (i == 0) ? negReservoir : negTemporalReservoir;
                        PathSampleRecord negXi = (i == 0) ? negSelectedPath : multiPointReSTIR.negPaths[curReservoir.pathID];
                        PathSampleRecord negYi;
                        float negJacobianToJ;
                        #if SHIFT_MAPPING_TYPE == 2
                            float2 negYiBaseFracPixelLoc = (i == 0) ? prevFracPixel : float2(pixel);
                            negXi.pathRcInfo.fracPixelLoc = (i == 0) ? negXi.pathRcInfo.fracPixelLoc : multiPointReSTIR.negTemporalFracPixelLocs[pixel];
                        #else
                            float2 negYiBaseFracPixelLoc = float2(negPrevBasePixel) + float2(0.5f);
                        #endif
                        float2 negYiShiftFracPixelLoc = negYiBaseFracPixelLoc + shift;
                        float2 negYiBasePixel = (negXi.index == 0) ? negYiBaseFracPixelLoc : negYiShiftFracPixelLoc;
                        float3 negYiBaseIntegrand = multiPointReSTIR.gPathTracer.areaReSTIRShiftMapping(negXi,
                                                                                   negYiBasePixel,
                                                                                   (i == 0) ? true : false,
                                                                                   rtq,
                                                                                   negYi,
                                                                                   negJacobianToJ).xyz;
                        negYi.index = negXi.index;
                        float3 integrand1 = multiPointReSTIR.gPathTracer.shiftAndComputeGPath( negYi,
                                                    negYiBaseFracPixelLoc,
                                                    negYiBaseIntegrand,
                                                    negYiShiftFracPixelLoc, rtq,
                                                    i == 0 ? true : false ) .xyz;
                        negJacobianToJ = (i == 0) ? negJacobianToJ : SafeNumber(1.0f / negJacobianToJ);
                        float3 integrand0 = curReservoir.integrand;
                        float c0 = (i == 0) ? 1.0f : negTemporalReservoir.C;
                        float c1 = (i == 0) ? negTemporalReservoir.C : 1.0f;
                        float w1 = c0 * negTargetFunc(integrand0) * (i == 0 ? 1.0f : negJacobianToJ);
                        float w2 = c1 * negTargetFunc(integrand1) * (i == 1 ? 1.0f : negJacobianToJ);
                        float negMisWeight = (w1 + w2 > 0.0f) ? w1 / (w1 + w2) : 0.0f;
                        if (!shouldReuseBasedOnJacobian(negXi.innerJacobian, negYi.innerJacobian))
                        {
                            negMisWeight = i == 0 ? 1.0f : 0.0f;
                        }
                        if (!curReservoir.isNullSample)
                        {
                            float negRisSelectU = sampleNext1D(risSg);
                            float3 negYiIntegrand = (i == 0) ? integrand0 : integrand1;
                            bool isSelected = negTmpReservoir.addTemporalSample( finalSelectedPathBufferIndex,
                                                                                 negYiIntegrand,
                                                                                 negTargetFunc(negYiIntegrand),
                                                                                 i == 0 ? 1.0f : SafeNumber(1.0f / negJacobianToJ),
                                                                                 negMisWeight,
                                                                                 c0,
                                                                                 curReservoir.W,
                                                                                 negRisSelectU );
                            if (i == 1 && isSelected)
                            {
                                negSelectedPath = negYi;
                            }
                        }
                    }
                    // {
                    //     PathReservoir negCurReservoir = negReservoir;
                    //     PathSampleRecord negYi;
                    //     float negJacobianToJ;
                    // #if SHIFT_MAPPING_TYPE == 2
                    //     float2 negYiBaseFracPixelLoc = prevFracPixel;
                    // #else
                    //     float2 negYiBaseFracPixelLoc = float2(negPrevBasePixel) + float2(0.5f);
                    // #endif
                    //     float2 negYiShiftFracPixelLoc = negYiBaseFracPixelLoc + shift;
                    //     float2 negYiBasePixel = (negSelectedPath.index == 0) ? negYiBaseFracPixelLoc : negYiShiftFracPixelLoc;
                    //     float3 negYiBaseIntegrand = gPathTracer.areaReSTIRShiftMapping(negSelectedPath,
                    //                                                                    negYiBasePixel,
                    //                                                                    true,
                    //                                                                    rtq,
                    //                                                                    negYi,
                    //                                                                    negJacobianToJ).xyz;
                    //     negYi.index = negSelectedPath.index;
                    //     float3 negYiIntegrand = gPathTracer.shiftAndComputeGPath(negYi,
                    //                                                              negYiBaseFracPixelLoc,
                    //                                                              negYiBaseIntegrand,
                    //                                                              negYiShiftFracPixelLoc,
                    //                                                              rtq,
                    //                                                              true).xyz;
                    //     float3 negXiIntegrand = negCurReservoir.integrand;
                    //     float negDenom = negTargetFunc(negXiIntegrand) +
                    //                      negTargetFunc(negYiIntegrand) * negTemporalReservoir.C * negJacobianToJ;
                    //     float negNum = negTargetFunc(negYiIntegrand);

                    //     float negMisWeight = (negDenom > 0.0f &&
                    //                           shouldReuseBasedOnJacobian(negSelectedPath.innerJacobian, negYi.innerJacobian)) ? negNum / negDenom : 1.0f;
                    //     bool isSelected = negTmpReservoir.addTemporalSample( selectedPathBufferIndex,
                    //         negXiIntegrand,
                    //         negTargetFunc(negXiIntegrand),
                    //         1.0f,
                    //         negMisWeight,
                    //         1.0f,
                    //         negCurReservoir.W,
                    //         sampleNext1D(risSg) );
                    // }

                    // {
                    //     PathSampleRecord negXi = multiPointReSTIR.negPaths[negTemporalReservoir.pathID];
                    //     PathSampleRecord negYi;
                    //     float negJacobianToCanonical;
                    // #if SHIFT_MAPPING_TYPE == 2
                    //     float2 negYiBaseFracPixelLoc = float2(pixel);
                    //     negXi.pathRcInfo.fracPixelLoc = multiPointReSTIR.negTemporalFracPixelLocs[pixel];
                    // #else
                    //     float2 negYiBaseFracPixelLoc = float2(pixel) + float2(0.5f);
                    // #endif
                    //     float2 negYiShiftFracPixelLoc = negYiBaseFracPixelLoc + shift;
                    //     float2 negYiBasePixel = (negXi.index == 0) ? negYiBaseFracPixelLoc : negYiShiftFracPixelLoc;
                    //     float3 negYiBaseIntegrand = multiPointReSTIR.gPathTracer.areaReSTIRShiftMapping(negXi,
                    //                                                                    negYiBasePixel,
                    //                                                                    false,
                    //                                                                    rtq,
                    //                                                                    negYi,
                    //                                                                    negJacobianToCanonical).xyz;
                    //     negYi.index = negXi.index;
                    //     float3 negYiIntegrand = multiPointReSTIR.gPathTracer.shiftAndComputeGPath(negYi,
                    //                                                           negYiBaseFracPixelLoc,
                    //                                                           negYiBaseIntegrand,
                    //                                                           negYiShiftFracPixelLoc,
                    //                                                           rtq,
                    //                                                           false).xyz;
                    //     float3 negXiIntegrand = negTemporalReservoir.integrand;
                    //     float negDenom = SafeNumber(negTargetFunc(negXiIntegrand) * negTemporalReservoir.C / negJacobianToCanonical) +
                    //                      negTargetFunc(negYiIntegrand);
                    //     float negNum = SafeNumber(negTargetFunc(negXiIntegrand) * negTemporalReservoir.C / negJacobianToCanonical);
                    //     float negMisWeight = (negDenom > 0.0f && shouldReuseBasedOnJacobian(negXi.innerJacobian, negYi.innerJacobian)) ? negNum / negDenom : 0.0f;
                    //     float negRisSelectU = sampleNext1D(risSg);
                    //     {
                    //         bool isSelected = negTmpReservoir.addTemporalSample( selectedPathBufferIndex,
                    //                                                              negYiIntegrand,
                    //                                                              negTargetFunc(negYiIntegrand),
                    //                                                              negJacobianToCanonical,
                    //                                                              negMisWeight,
                    //                                                              negTemporalReservoir.C,
                    //                                                              negTemporalReservoir.W,
                    //                                                              negRisSelectU );
                    //         negSelectedPath = isSelected ? negYi : negSelectedPath;
                    //     }
                    // }
                    negTmpReservoir.finalizeTemporalReuse(multiPointReSTIR.gPathTracer.params.historyLength);
                    negReservoir = negTmpReservoir;
                }
            }

            // Write paths and reservoirs to the buffer
            {
                multiPointReSTIR.negReservoirs[finalSelectedPathBufferIndex] = negReservoir;
                multiPointReSTIR.negPaths[finalSelectedPathBufferIndex] = negSelectedPath;
            }
        }
    }
};

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    printSetPixel(dispatchThreadId.xy);
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel < gPathTracer.params.frameDim))
    {
        gGradientIntegrator.run(pixel);
    }
}
