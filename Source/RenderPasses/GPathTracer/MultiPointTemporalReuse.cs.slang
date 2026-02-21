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
StructuredBuffer<float4> shiftedPathsRadiance;
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
        SampleGenerator risSg = SampleGenerator(pixel, gPathTracer.params.seed * (kNumInitialSamples + 1));
        uint basePathBufferIndex = getBufferOffset(pixel, frameDim, kNumInitialSamples);
        uint2 shiftPixel = pixel + shift;
        uint shiftPathBufferIndex = getBufferOffset(shiftPixel, frameDim, kNumInitialSamples);
        uint baseBufferIndex = min(basePathBufferIndex, shiftPathBufferIndex);

        uint basePixelBufferIndex = getBufferOffset(pixel, frameDim);
        uint shiftPixelBufferIndex = getBufferOffset(shiftPixel, frameDim);

        // bool haveSpatialNeighbour = !all(neighbourPixels[pixel] == pixel) && kNumSpatialNeighbours > 0;
        uint selectedPathBufferIndex = getBufferOffset(pixel, frameDim, pathsPerPixel);

        uint negSelectedPathBufferIndex = getBufferOffset(pixel, frameDim, pathsPerPixel);

        if (isInFrame(pixel, frameDim) && isInFrame(shiftPixel, frameDim))
        {
            RTQueryImpl rtq;
            float posSelectedPathInnerJacobian = 0.0f;
            uint posSelectedPathIndex;
            float negSelectedPathInnerJacobian = 0.0f;
            uint negSelectedPathIndex;
            PathReservoir posReservoir; posReservoir.init();
            PathReservoir negReservoir; negReservoir.init();
            PathSampleRecord posSelectedPath;
            PathSampleRecord negSelectedPath;
            // Initial Samples
            {
                for (uint j = 0; j < 2; ++j)
                {
                    uint2 curBasePixel = (j == 0) ? pixel : shiftPixel;
                    uint curBasePathBufferIndex = (j == 0) ? basePathBufferIndex : shiftPathBufferIndex;
                    float s = (j == 0) ? 1.0f : -1.0f;
                    for (uint i = 0; i < kNumInitialSamples; ++i)
                    {
                        float pathW = multiPointReSTIR.basePathsRadiance[curBasePathBufferIndex + i].w;
                        float3 pathIntegral = multiPointReSTIR.basePathsRadiance[curBasePathBufferIndex + i].xyz;
                        float3 shiftPathIntegral = shiftedPathsRadiance[2*(curBasePathBufferIndex + i) + j].xyz;
                        float innerJacobian = shiftedPathsRadiance[2*(curBasePathBufferIndex + i) + j].w;
                        float misWeight = targetFunc(pathIntegral) / (targetFunc(pathIntegral) + targetFunc(shiftPathIntegral) * innerJacobian);
                        float3 signedDiff = s * misWeight * (shiftPathIntegral * innerJacobian - pathIntegral);

                        float mi = 1.0f / kNumInitialSamples;
                        float posPHatI = posTargetFunc(signedDiff.xyz);
                        float posRisSelectU = sampleNext1D(risSg);
                        bool posSelected = posReservoir.addCandidateSample(curBasePathBufferIndex + i,
                                                                      signedDiff.xyz,
                                                                      posPHatI,
                                                                      mi,
                                                                      pathW,
                                                                      posRisSelectU );
                        posSelectedPathInnerJacobian = posSelected ? innerJacobian : posSelectedPathInnerJacobian;
                        posSelectedPathIndex = posSelected ? j : posSelectedPathIndex;

                        float negPHatI = negTargetFunc(signedDiff.xyz);
                        float negRisSelectU = sampleNext1D(risSg);
                        bool negSelected = negReservoir.addCandidateSample(curBasePathBufferIndex + i,
                                                                      signedDiff.xyz,
                                                                      negPHatI,
                                                                      mi,
                                                                      pathW,
                                                                      negRisSelectU );
                        negSelectedPathInnerJacobian = negSelected ? innerJacobian : negSelectedPathInnerJacobian;
                        negSelectedPathIndex = negSelected ? j : negSelectedPathIndex;
                    }
                }

                // for (uint i = 0; i < kNumInitialSamples; ++i)
                // {
                //     uint baseIndex = frameDim.x * frameDim.y * i;
                //     float pathW = multiPointReSTIR.basePathsRadiance[shiftPathBufferIndex + i].w;
                //     float3 pathIntegral = multiPointReSTIR.basePathsRadiance[shiftPathBufferIndex + i].xyz;
                //     float3 shiftPathIntegral = shiftedPathsRadiance[2*(shiftPathBufferIndex + i) + 1].xyz;
                //     float innerJacobian = shiftedPathsRadiance[2*(shiftPathBufferIndex + i) + 1].w;
                //     float misWeight = targetFunc(pathIntegral) / (targetFunc(pathIntegral) + targetFunc(shiftPathIntegral) * innerJacobian);
                //     float3 signedDiff = -misWeight * (shiftPathIntegral * innerJacobian - pathIntegral);

                //     float mi = 1.0f / kNumInitialSamples;
                //     float posPHatI = posTargetFunc(signedDiff.xyz);
                //     float posRisSelectU = sampleNext1D(risSg);
                //     bool posSelected = posReservoir.addCandidateSample(shiftPathBufferIndex + i,
                //                                                   signedDiff.xyz,
                //                                                   posPHatI,
                //                                                   mi,
                //                                                   pathW,
                //                                                   posRisSelectU );
                //     posSelectedPathInnerJacobian = posSelected ? innerJacobian : posSelectedPathInnerJacobian;
                //     posSelectedPathIndex = posSelected ? 1 : posSelectedPathIndex;

                //     float negPHatI = negTargetFunc(signedDiff.xyz);
                //     float negRisSelectU = sampleNext1D(risSg);
                //     bool negSelected = negReservoir.addCandidateSample(shiftPathBufferIndex + i,
                //                                                   signedDiff.xyz,
                //                                                   negPHatI,
                //                                                   mi,
                //                                                   pathW,
                //                                                   negRisSelectU );
                //     negSelectedPathInnerJacobian = negSelected ? innerJacobian : negSelectedPathInnerJacobian;
                //     negSelectedPathIndex = negSelected ? 1 : negSelectedPathIndex;
                // }

                posReservoir.finalizeInitialCandidates();
                negReservoir.finalizeInitialCandidates();
                posSelectedPath = multiPointReSTIR.basePaths[posReservoir.pathID];
                posSelectedPath.index = posSelectedPathIndex;
                posSelectedPath.innerJacobian = posSelectedPathInnerJacobian;
                negSelectedPath = multiPointReSTIR.basePaths[negReservoir.pathID];
                negSelectedPath.index = negSelectedPathIndex;
                negSelectedPath.innerJacobian = negSelectedPathInnerJacobian;
                posReservoir.pathID = selectedPathBufferIndex;
                negReservoir.pathID = negSelectedPathBufferIndex;
            }
            // Temporal Reuse
            PathReservoir posTemporalReservoir = multiPointReSTIR.posSampledTemporalReservoirs[basePixelBufferIndex];
            bool posFoundNeighbour = !posTemporalReservoir.isNullSample;
            if (kUseTemporalReuse && posFoundNeighbour)
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

                if (isInFrame(shiftPixel, frameDim) && posFoundNeighbour)
                {
                    RTQueryImpl rtq;
                    PathReservoir posTmpReservoir;
                    posTmpReservoir.init();
                    {
                        for (int i = 0; i < 2; ++i)
                        {
                            PathReservoir curReservoir = (i == 0) ? posReservoir : posTemporalReservoir;
                            PathSampleRecord posXi = (i == 0) ? posSelectedPath : multiPointReSTIR.posPaths[curReservoir.pathID];
                            PathSampleRecord posYi;
                            float posJacobianToJ;
                            #if SHIFT_MAPPING_TYPE == 2
                                float2 posYiBaseFracPixelLoc = (i == 0) ? prevFracPixel : float2(pixel);
                                posXi.pathRcInfo.fracPixelLoc = (i == 0) ? posXi.pathRcInfo.fracPixelLoc : multiPointReSTIR.posTemporalFracPixelLocs[pixel];
                            #else
                                float2 posYiBaseFracPixelLoc = float2(posPrevBasePixel) + float2(0.5f);
                            #endif
                            float2 posYiShiftFracPixelLoc = posYiBaseFracPixelLoc + shift;
                            float2 posYiBasePixel = (posXi.index == 0) ? posYiBaseFracPixelLoc : posYiShiftFracPixelLoc;
                            float3 posYiBaseIntegrand = multiPointReSTIR.gPathTracer.areaReSTIRShiftMapping(posXi,
                                                                                       posYiBasePixel,
                                                                                       (i == 0) ? true : false,
                                                                                       rtq,
                                                                                       posYi,
                                                                                       posJacobianToJ).xyz;
                            posYi.index = posXi.index;
                            float3 integrand1 = multiPointReSTIR.gPathTracer.shiftAndComputeGPath( posYi,
                                                        posYiBaseFracPixelLoc,
                                                        posYiBaseIntegrand,
                                                        posYiShiftFracPixelLoc, rtq,
                                                        i == 0 ? true : false ) .xyz;
                            posJacobianToJ = (i == 0) ? posJacobianToJ : SafeNumber(1.0f / posJacobianToJ);
                            float3 integrand0 = curReservoir.integrand;
                            float3 posYiIntegrand = (i == 0) ? integrand0 : integrand1;
                            float c0 = (i == 0) ? 1.0f : posTemporalReservoir.C;
                            float c1 = (i == 0) ? posTemporalReservoir.C : 1.0f;
                            float w0 = c0 * posTargetFunc(integrand0) * (i == 0 ? 1.0f : posJacobianToJ);
                            float w1 = c1 * posTargetFunc(integrand1) * (i == 1 ? 1.0f : posJacobianToJ);
                            float posMisWeight = (w0 + w1 > 0.0f) ? w0 / (w0 + w1) : 0.0f;
                            if (!shouldReuseBasedOnJacobian(posXi.innerJacobian, posYi.innerJacobian))
                            {
                                posMisWeight = i == 0 ? 1.0f : 0.0f;
                            }
                            float posRisSelectU = sampleNext1D(risSg);
                            if (!curReservoir.isNullSample)
                            {
                                bool isSelected = posTmpReservoir.addTemporalSample( selectedPathBufferIndex,
                                                                                     posYiIntegrand,
                                                                                     posTargetFunc(posYiIntegrand),
                                                                                     i == 0 ? 1.0f : SafeNumber(1.0f / posJacobianToJ),
                                                                                     posMisWeight,
                                                                                     c0,
                                                                                     curReservoir.W,
                                                                                     posRisSelectU );
                                if (i == 1 && isSelected)
                                {
                                    posSelectedPath = posYi;
                                }
                            }
                        }
                    //     PathReservoir posCurReservoir = posReservoir;
                    //     PathSampleRecord posXi = posSelectedPath;
                    //     PathSampleRecord posYi;
                    //     float posJacobianToJ;
                    // #if SHIFT_MAPPING_TYPE == 2
                    //     float2 posXjBaseFracPixelLoc = prevFracPixel;
                    // #else
                    //     float2 posXjBaseFracPixelLoc = float2(posPrevBasePixel) + float2(0.5f);
                    // #endif
                    //     float2 posXjShiftFracPixelLoc = posXjBaseFracPixelLoc + shift;
                    //     float2 posYiBasePixel = (posXi.index == 0) ? posXjBaseFracPixelLoc : posXjShiftFracPixelLoc;
                    //     float3 posYiBaseIntegrand = gPathTracer.areaReSTIRShiftMapping(posXi,
                    //                                                                    posYiBasePixel,
                    //                                                                    true,
                    //                                                                    rtq,
                    //                                                                    posYi,
                    //                                                                    posJacobianToJ).xyz;
                    //     posYi.index = posXi.index;
                    //     float3 posYiIntegrand = gPathTracer.shiftAndComputeGPath(posYi,
                    //                                                              posXjBaseFracPixelLoc,
                    //                                                              posYiBaseIntegrand,
                    //                                                              posXjShiftFracPixelLoc,
                    //                                                              rtq,
                    //                                                              true).xyz;

                    //     float3 posXiIntegrand = posCurReservoir.integrand;
                    //     float posDenom = posTargetFunc(posXiIntegrand) + posTargetFunc(posYiIntegrand) * posTemporalReservoir.C * posJacobianToJ;
                    //                      // (shouldNegReuse ? negTargetFunc(posXjIntegrand) * negTemporalReservoir.C * posJacobianToJ : 0.0f);
                    //     float posNum = posTargetFunc(posXiIntegrand);
                    //     float posMisWeight = (posDenom > 0.0f &&
                    //                           shouldReuseBasedOnJacobian(posXi.innerJacobian, posYi.innerJacobian)) ? posNum / posDenom : 1.0f;

                    //     if (!posReservoir.isNullSample)
                    //     {
                    //         posTmpReservoir.addTemporalSample( selectedPathBufferIndex,
                    //             posXiIntegrand,
                    //             posTargetFunc(posXiIntegrand),
                    //             1.0f, posMisWeight, 1.0f,
                    //             posCurReservoir.W,
                    //             sampleNext1D(risSg) );
                    //     }
                    }

                    // {
                    //     PathSampleRecord posXi = multiPointReSTIR.posPaths[posTemporalReservoir.pathID];
                    //     PathSampleRecord posYi;
                    //     float posJacobianToCanonical;
                    // #if SHIFT_MAPPING_TYPE == 2
                    //     float2 posYiBaseFracPixelLoc =  float2(pixel);
                    //     posXi.pathRcInfo.fracPixelLoc = multiPointReSTIR.posTemporalFracPixelLocs[pixel];
                    // #else
                    //     float2 posYiBaseFracPixelLoc =  float2(pixel) + float2(0.5f);
                    // #endif
                    //     float2 posYiShiftFracPixelLoc =  posYiBaseFracPixelLoc + shift;
                    //     float2 posYiBasePixel = (posXi.index == 0) ? posYiBaseFracPixelLoc : posYiShiftFracPixelLoc;
                    //     float3 posYiBaseIntegrand = multiPointReSTIR.gPathTracer.areaReSTIRShiftMapping(posXi,
                    //                                                                    posYiBasePixel,
                    //                                                                    false,
                    //                                                                    rtq,
                    //                                                                    posYi,
                    //                                                                    posJacobianToCanonical).xyz;
                    //     posYi.index = posXi.index;
                    //     float3 posYiIntegrand = multiPointReSTIR.gPathTracer.shiftAndComputeGPath(posYi,
                    //                                                           posYiBaseFracPixelLoc,
                    //                                                           posYiBaseIntegrand,
                    //                                                           posYiShiftFracPixelLoc, rtq).xyz;
                    //     float3 posXiIntegrand = posTemporalReservoir.integrand;
                    //     float posDenom = SafeNumber(posTargetFunc(posXiIntegrand) * posTemporalReservoir.C / posJacobianToCanonical) +
                    //                      posTargetFunc(posYiIntegrand);
                    //     float posNum = SafeNumber(posTargetFunc(posXiIntegrand) * posTemporalReservoir.C / posJacobianToCanonical);
                    //     float posMisWeight = (posDenom > 0.0f && shouldReuseBasedOnJacobian(posXi.innerJacobian, posYi.innerJacobian)) ? posNum / posDenom : 0.0f;
                    //     float posRisSelectU = sampleNext1D(risSg);
                    //     {
                    //         bool isSelected = posTmpReservoir.addTemporalSample( selectedPathBufferIndex,
                    //                                                              posYiIntegrand,
                    //                                                              posTargetFunc(posYiIntegrand),
                    //                                                              posJacobianToCanonical,
                    //                                                              posMisWeight,
                    //                                                              posTemporalReservoir.C,
                    //                                                              posTemporalReservoir.W,
                    //                                                              posRisSelectU );
                    //         posSelectedPath = isSelected ? posYi : posSelectedPath;
                    //     }
                    // }
                    posTmpReservoir.finalizeTemporalReuse(multiPointReSTIR.gPathTracer.params.historyLength);
                    posReservoir = posTmpReservoir;
                        }

                        // if (negFoundNeighbour)
                        // {
                        //     PathReservoir negTmpReservoir;
                        //     negTmpReservoir.init();
                        //     {
                        //         PathReservoir negCurReservoir = negReservoir;
                        //         PathSampleRecord negXi = negSelectedPath;
                        //         PathSampleRecord negYi;
                        //         float negJacobianToJ;
                        //     #if SHIFT_MAPPING_TYPE == 2
                        //         float2 negYiBaseFracPixelLoc = prevFracPixel;
                        //     #else
                        //         float2 negYiBaseFracPixelLoc = float2(negPrevBasePixel) + float2(0.5f);
                        //     #endif
                        //         float2 negYiShiftFracPixelLoc = negYiBaseFracPixelLoc + shift;
                        //         float2 negYiBasePixel = (negXi.index == 0) ? negYiBaseFracPixelLoc : negYiShiftFracPixelLoc;
                        //         float3 negYiBaseIntegrand = gPathTracer.areaReSTIRShiftMapping(negXi,
                        //                                                                        negYiBasePixel,
                        //                                                                        true,
                        //                                                                        rtq,
                        //                                                                        negYi,
                        //                                                                        negJacobianToJ).xyz;
                        //         negYi.index = negXi.index;
                        //         float3 negYiIntegrand = gPathTracer.shiftAndComputeGPath(negYi,
                        //                                                                  negYiBaseFracPixelLoc,
                        //                                                                  negYiBaseIntegrand,
                        //                                                                  negYiShiftFracPixelLoc,
                        //                                                                  rtq,
                        //                                                                  true).xyz;
                        //         float3 negXiIntegrand = negCurReservoir.integrand;
                        //         float negDenom = negTargetFunc(negXiIntegrand) +
                        //                          negTargetFunc(negYiIntegrand) * negTemporalReservoir.C * negJacobianToJ;
                        //         float negNum = negTargetFunc(negYiIntegrand);

                        //         float negMisWeight = (negDenom > 0.0f &&
                        //                               shouldReuseBasedOnJacobian(negXi.innerJacobian, negYi.innerJacobian)) ? negNum / negDenom : 1.0f;
                        //         negTmpReservoir.addTemporalSample( selectedPathBufferIndex,
                        //                                            negXiIntegrand,
                        //                                            negTargetFunc(negXiIntegrand),
                        //                                            1.0f,
                        //                                            negMisWeight,
                        //                                            1.0f,
                        //                                            negCurReservoir.W,
                        //                                            sampleNext1D(risSg) );
                        //     }

                        //     {
                        //         PathSampleRecord negXi = multiPointReSTIR.negPaths[negTemporalReservoir.pathID];
                        //         PathSampleRecord negYi;
                        //         float negJacobianToCanonical;
                        //     #if SHIFT_MAPPING_TYPE == 2
                        //         float2 negYiBaseFracPixelLoc = float2(pixel);
                        //         negXi.pathRcInfo.fracPixelLoc = multiPointReSTIR.negTemporalFracPixelLocs[pixel];
                        //     #else
                        //         float2 negYiBaseFracPixelLoc = float2(pixel) + float2(0.5f);
                        //     #endif
                        //         float2 negYiShiftFracPixelLoc = negYiBaseFracPixelLoc + shift;
                        //         float2 negYiBasePixel = (negXi.index == 0) ? negYiBaseFracPixelLoc : negYiShiftFracPixelLoc;
                        //         float3 negYiBaseIntegrand = multiPointReSTIR.gPathTracer.areaReSTIRShiftMapping(negXi,
                        //                                                                        negYiBasePixel,
                        //                                                                        false,
                        //                                                                        rtq,
                        //                                                                        negYi,
                        //                                                                        negJacobianToCanonical).xyz;
                        //         negYi.index = negXi.index;
                        //         float3 negYiIntegrand = multiPointReSTIR.gPathTracer.shiftAndComputeGPath(negYi,
                        //                                                               negYiBaseFracPixelLoc,
                        //                                                               negYiBaseIntegrand,
                        //                                                               negYiShiftFracPixelLoc,
                        //                                                               rtq,
                        //                                                               false).xyz;
                        //         float3 negXiIntegrand = negTemporalReservoir.integrand;
                        //         float negDenom = SafeNumber(negTargetFunc(negXiIntegrand) * negTemporalReservoir.C / negJacobianToCanonical) +
                        //                          negTargetFunc(negYiIntegrand);
                        //         float negNum = SafeNumber(negTargetFunc(negXiIntegrand) * negTemporalReservoir.C / negJacobianToCanonical);
                        //         float negMisWeight = (negDenom > 0.0f && shouldReuseBasedOnJacobian(negXi.innerJacobian, negYi.innerJacobian)) ? negNum / negDenom : 0.0f;
                        //         float negRisSelectU = sampleNext1D(risSg);
                        //         {
                        //             bool isSelected = negTmpReservoir.addTemporalSample( selectedPathBufferIndex,
                        //                                                                  negYiIntegrand,
                        //                                                                  negTargetFunc(negYiIntegrand),
                        //                                                                  negJacobianToCanonical,
                        //                                                                  negMisWeight,
                        //                                                                  negTemporalReservoir.C,
                        //                                                                  negTemporalReservoir.W,
                        //                                                                  negRisSelectU );
                        //             negSelectedPath = isSelected ? negYi : negSelectedPath;
                        //         }
                        //     }
                        //     negTmpReservoir.finalizeTemporalReuse(multiPointReSTIR.gPathTracer.params.historyLength);
                        //     negReservoir = negTmpReservoir;
                        // }
            }

            // Write paths and reservoirs to the buffer
            {
                multiPointReSTIR.posReservoirs[selectedPathBufferIndex] = posReservoir;
                multiPointReSTIR.negReservoirs[negSelectedPathBufferIndex] = negReservoir;
                multiPointReSTIR.posPaths[selectedPathBufferIndex] = posSelectedPath;
                multiPointReSTIR.negPaths[negSelectedPathBufferIndex] = negSelectedPath;
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
