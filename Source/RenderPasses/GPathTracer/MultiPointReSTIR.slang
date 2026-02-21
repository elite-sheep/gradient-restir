// Copyright @yucwang 2024

import Scene.RaytracingInline;
import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Debug.PixelDebug;

import CameraUtils;
import DiffIntegratorInterface;
import GPathTracer;
import PathShift;
import PathState;
import RISDataTypes;
import StaticParams;
import Utils;

struct MultiPointReSTIR : IDiffIntegrator
{
    ParameterBlock<GPathTracer> gPathTracer;
    RWStructuredBuffer<PathReservoir> posReservoirs;
    RWStructuredBuffer<PathReservoir> negReservoirs;
    RWStructuredBuffer<PathSampleRecord> posPaths;
    RWStructuredBuffer<PathSampleRecord> negPaths;
    StructuredBuffer<PathSampleRecord> basePaths;
    StructuredBuffer<float4> basePathsRadiance;

    RWStructuredBuffer<PathReservoir> posSampledTemporalReservoirs;
    RWStructuredBuffer<PathReservoir> negSampledTemporalReservoirs;
    RWTexture2D<float2> posTemporalFracPixelLocs;
    RWTexture2D<float2> negTemporalFracPixelLocs;

    Texture1D<float2> neighborOffsets;
    Texture2D<float> primaryDepth;
    Texture2D<float3> primaryEmission;
    Texture2D<float4> primaryAlbedo;
    Texture2D<float4> primaryNormalAndDepth;

    uint pathsPerPixel;
    uint shiftDir;

    bool shouldSpatialReuse(const uint2 pixel, const uint2 shiftPixel)
    {
        float4 albedoDiff = primaryAlbedo[shiftPixel] - primaryAlbedo[pixel];
        bool isAlbedoFine = maxElement(albedoDiff.xyz) > 0.05f;

        float normalDot = dot(primaryNormalAndDepth[shiftPixel].xyz, primaryNormalAndDepth[pixel].xyz);
        bool isNormalFine = abs(normalDot) < 0.7f;

        float depthDiff = primaryDepth[shiftPixel] - primaryDepth[pixel];
        bool isDepthFine = abs(depthDiff) / (primaryDepth[pixel] + 1e-6) > 0.01f;

        return isAlbedoFine || isNormalFine || isDepthFine;
    }

    int2 getNextNeighborPixel(const uint startIndex, uint2 pixel, int i)
    {
        float gGatherRadius = gPathTracer.params.spatialGatherRadius;
        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * gGatherRadius);
        // int s = neighborOffsets[neighborIndex].x > 0.0f ? 1 : -1;
        // int2 shiftDirVec = shiftDir == 0 ? int2(0, neighborOffsets[neighborIndex].y * gGatherRadius) :
        //                                    int2(neighborOffsets[neighborIndex].x * gGatherRadius, 0);

        return neighborPixel;
    }

    int2 getNeibourPixel(const uint2 pixel, const uint2 shift, const uint startIndex)
    {
        int2 neighborPixel = pixel;
        for (int i = 0; i < 1024; ++i)
        {
            int2 curXiPixel = getNextNeighborPixel(startIndex, pixel, i);
            if (shouldSpatialReuse(curXiPixel, curXiPixel + shift) &&
                !shouldSpatialReuse(pixel, curXiPixel))
            {
                neighborPixel = curXiPixel;
                break;
            }
        }

        return neighborPixel;
    }

    bool shouldSpatialReuse(int iNeighbour, uint neighbourValidMask)
    {
        if (iNeighbour == -1)
            return true;
        else
            return getBit(neighbourValidMask, iNeighbour);
    }

    bool isGeometrySimilar(const VBuffer vBuffer1, const VBuffer vBuffer2, bool allowInValid = false)
    {
        if (allowInValid && !vBuffer1.primaryHitInfo.isValid() && !vBuffer2.primaryHitInfo.isValid())
            return true;
        return vBuffer1.primaryHitInfo.isValid() && vBuffer2.primaryHitInfo.isValid() &&
               gPathTracer.isValidGeometry(vBuffer1, vBuffer2);
    }

    float4 evaluate(uint2 basePixel, uint2 shift)
    {
        uint bufferIndex = getBufferOffset(basePixel, gPathTracer.params.frameDim, pathsPerPixel) + pathsPerPixel - 1;
        uint2 shiftPixel = basePixel + shift;

        float3 finalColor = float3(0.0f);
        if (isInFrame(basePixel, gPathTracer.params.frameDim) && isInFrame(shiftPixel, gPathTracer.params.frameDim))
        {
            float3 emissionDiff = primaryEmission[shiftPixel] - primaryEmission[basePixel];
            finalColor = emissionDiff +
                         posReservoirs[bufferIndex].W * max(posReservoirs[bufferIndex].integrand, float3(0.0f)) +
                         negReservoirs[bufferIndex].W * min(negReservoirs[bufferIndex].integrand, float3(0.0f));
        }

        return float4(finalColor, 0.0f);
    }
}

void MultiPointSampleInitialCandidates(MultiPointReSTIR multiPointReSTIR, uint2 basePixel, uint2 shift)
{
    // SampleGenerator risSg = SampleGenerator(basePixel,
    //                                 multiPointReSTIR.gPathTracer.params.seed * (kNumInitialSamples + 1));
    // uint baseBufferIndex = getBufferOffset(basePixel, multiPointReSTIR.gPathTracer.params.frameDim);
    // uint2 shiftPixel = basePixel + shift;
    // uint shiftBufferIndex = getBufferOffset(shiftPixel, multiPointReSTIR.gPathTracer.params.frameDim);
    // uint2 frameDim = multiPointReSTIR.gPathTracer.params.frameDim;
    // // print("frameDim:", frameDim);
    // // print("basePixel:", basePixel);
    // // print("shiftPixel:", shiftPixel);
    // if (isInFrame(basePixel, multiPointReSTIR.gPathTracer.params.frameDim) &&
    //     isInFrame(shiftPixel, multiPointReSTIR.gPathTracer.params.frameDim))
    // {
    //     RTQueryImpl rtq;
    //     for (uint i = 0; i < kNumInitialSamples; ++i)
    //     {
    //         uint baseIndex = frameDim.x * frameDim.y * i;
    //         PathSampleRecord path = multiPointReSTIR.basePaths[baseIndex + baseBufferIndex];
    //         path.index = 0;
    //         float3 signedDiff = multiPointReSTIR.gPathTracer.shiftAndComputeGPath( path,
    //                                             basePixel,
    //                                             multiPointReSTIR.gPathTracer.vbuffer[baseBufferIndex],
    //                                             shiftPixel,
    //                                             multiPointReSTIR.gPathTracer.vbuffer[shiftBufferIndex], rtq );

    //         // print("basePixel:", path.sampleBaseFracPixelLoc);
    //         // print("shiftPixel:", path.sampleShiftFracPixelLoc);
    //         // print("baseIndex:", baseBufferIndex);
    //         // print("shiftBufferIndex:", shiftBufferIndex);

    //         float mi = 1.0f / kNumInitialSamples;
    //         float posPHatI = posTargetFunc(signedDiff.xyz);
    //         float posRisSelectU = sampleNext1D(risSg);
    //         multiPointReSTIR.posReservoirs[baseBufferIndex].addCandidateSample(path,
    //                                                       signedDiff.xyz,
    //                                                       posPHatI,
    //                                                       mi,
    //                                                       path.pathW,
    //                                                       posRisSelectU );

    //         float negPHatI = negTargetFunc(signedDiff.xyz);
    //         float negRisSelectU = sampleNext1D(risSg);
    //         multiPointReSTIR.negReservoirs[baseBufferIndex].addCandidateSample(path,
    //                                                       signedDiff.xyz,
    //                                                       negPHatI,
    //                                                       mi,
    //                                                       path.pathW,
    //                                                       negRisSelectU );
    //     }

    //     for (uint i = 0; i < kNumInitialSamples; ++i)
    //     {
    //         uint baseIndex = frameDim.x * frameDim.y * i;
    //         PathSampleRecord path = multiPointReSTIR.basePaths[baseIndex + shiftBufferIndex];
    //         path.index = 1;
    //         float3 signedDiff = multiPointReSTIR.gPathTracer.shiftAndComputeGPath( path,
    //             basePixel,
    //             multiPointReSTIR.gPathTracer.vbuffer[baseBufferIndex],
    //             shiftPixel,
    //             multiPointReSTIR.gPathTracer.vbuffer[shiftBufferIndex], rtq );

    //         float mi = 1.0f / kNumInitialSamples;
    //         float posPHatI = posTargetFunc(signedDiff.xyz);
    //         float posRisSelectU = sampleNext1D(risSg);
    //         multiPointReSTIR.posReservoirs[baseBufferIndex].addCandidateSample(path,
    //                                                       signedDiff.xyz,
    //                                                       posPHatI,
    //                                                       mi,
    //                                                       path.pathW,
    //                                                       posRisSelectU);

    //         float negPHatI = negTargetFunc(signedDiff.xyz);
    //         float negRisSelectU = sampleNext1D(risSg);
    //         multiPointReSTIR.negReservoirs[baseBufferIndex].addCandidateSample( path,
    //                                                       signedDiff.xyz,
    //                                                       negPHatI,
    //                                                       mi,
    //                                                       path.pathW,
    //                                                       negRisSelectU );
    //     }
    //     print("pos.W:", multiPointReSTIR.posReservoirs[baseBufferIndex].W);
    //     print("pos.sampleBasePixel:", multiPointReSTIR.posReservoirs[baseBufferIndex].pathSampleRecord.sampleBaseFracPixelLoc);
    //     print("neg.sampleBasePixel:", multiPointReSTIR.negReservoirs[baseBufferIndex].pathSampleRecord.sampleBaseFracPixelLoc);

    //     multiPointReSTIR.posReservoirs[baseBufferIndex].finalizeInitialCandidates();
    //     multiPointReSTIR.negReservoirs[baseBufferIndex].finalizeInitialCandidates();
    // }
}

void MultiPointTemporalReuse(MultiPointReSTIR multiPointReSTIR, uint2 basePixel, uint2 shift)
{
//     SampleGenerator risSg = SampleGenerator(basePixel,
//                                     multiPointReSTIR.gPathTracer.params.seed * (kNumInitialSamples + 3));
//     uint2 frameDim = multiPointReSTIR.gPathTracer.params.frameDim;
//     uint bufferIndex = getBufferOffset(basePixel, frameDim);
//     uint2 shiftPixel = basePixel + shift;
//     uint shiftBufferIndex = getBufferOffset(shiftPixel, frameDim);
//
//     multiPointReSTIR.posTmpReservoirs[bufferIndex] = multiPointReSTIR.posReservoirs[bufferIndex];
//     multiPointReSTIR.negTmpReservoirs[bufferIndex] = multiPointReSTIR.negReservoirs[bufferIndex];
//
//     float2 motionVector = (multiPointReSTIR.gPathTracer.motionVectors[basePixel]
//                             + multiPointReSTIR.gPathTracer.motionVectors[shiftPixel]) * 0.5f;
// #if SHIFT_MAPPING_TYPE == 2
//     if (multiPointReSTIR.primaryDepth[basePixel] < multiPointReSTIR.primaryDepth[shiftPixel])
//     {
//         motionVector = multiPointReSTIR.gPathTracer.motionVectors[basePixel];
//     }
//     else
//     {
//         motionVector = multiPointReSTIR.gPathTracer.motionVectors[shiftPixel];
//     }
//     float2 screenMotion = motionVector * float2(frameDim);
//     float2 prevFracCenter = float2(basePixel) + float2(0.5f) + screenMotion;
//     float2 prevFracPixel = prevFracCenter - float2(0.5f);
//     int2 posPrevBasePixel;
//     PathReservoir posTemporalReservoir;
//     bool posFoundNeighbour = multiPointReSTIR.lookForTemporalPosNeighbour(prevFracCenter, shift, frameDim,
//                                                 risSg, posTemporalReservoir, posPrevBasePixel);
//     int2 posPrevShiftPixel = posPrevBasePixel + shift;
//     uint posPrevBufferIndex = getBufferOffset(posPrevBasePixel, frameDim);
//     uint posPrevShiftBufferIndex = getBufferOffset(posPrevShiftPixel, frameDim);
//
//     int2 negPrevBasePixel;
//     PathReservoir negTemporalReservoir;
//     bool negFoundNeighbour = multiPointReSTIR.lookForTemporalNegNeighbour(prevFracCenter, shift,frameDim,
//                                                 risSg, negTemporalReservoir, negPrevBasePixel);
//     int2 negPrevShiftPixel = negPrevBasePixel + shift;
//     uint negPrevBufferIndex = getBufferOffset(negPrevBasePixel, frameDim);
//     uint negPrevShiftBufferIndex = getBufferOffset(negPrevShiftPixel, frameDim);
// #else
//     int2 prevBasePixel = basePixel;
//     prevBasePixel = int2(basePixel + motionVector * frameDim + sampleNext2D(risSg));
//     uint prevBufferIndex = getBufferOffset(prevBasePixel, frameDim);
//     int2 prevShiftPixel = prevBasePixel + shift;
//
//     int2 posPrevBasePixel = prevBasePixel;
//     int2 posPrevShiftPixel = prevShiftPixel;
//     uint posPrevBufferIndex = prevBufferIndex;
//     uint posPrevShiftBufferIndex = getBufferOffset(posPrevShiftPixel, frameDim);
//     PathReservoir posTemporalReservoir = multiPointReSTIR.temporalPosReservoirs[posPrevBufferIndex];
//     bool posFoundNeighbour = isInFrame(posPrevBasePixel, frameDim) && isInFrame(posPrevShiftPixel, frameDim);
//
//     int2 negPrevBasePixel = prevBasePixel;
//     int2 negPrevShiftPixel = prevShiftPixel;
//     uint negPrevBufferIndex = prevBufferIndex;
//     uint negPrevShiftBufferIndex = getBufferOffset(negPrevShiftPixel, frameDim);
//     PathReservoir negTemporalReservoir = multiPointReSTIR.temporalNegReservoirs[negPrevBufferIndex];
//     bool negFoundNeighbour = isInFrame(negPrevBasePixel, frameDim) && isInFrame(negPrevShiftPixel, frameDim);
// #endif
//
//     // int2 selectedPrevBasePixel = int2(100000, 100000);
//     // for (int i = -2; i <= 2; ++i)
//     // {
//     //     for (int j = -2; j <= 2; ++j)
//     //     {
//     //         int2 curPixel = prevBasePixel + int2(i, j);
//     //         if (isInFrame(curPixel, frameDim) && isInFrame(curPixel + shift, frameDim))
//     //         {
//     //             uint curOffset = getBufferOffset(curPixel, frameDim);
//     //             if (multiPointReSTIR.isGeometrySimilar( multiPointReSTIR.posReservoirs[bufferIndex].primaryHitInfo,
//     //                     multiPointReSTIR.temporalPosReservoirs[curOffset].primaryHitInfo, true) &&
//     //                 multiPointReSTIR.isGeometrySimilar( multiPointReSTIR.posReservoirs[bufferIndex].shiftPrimaryHitInfo,
//     //                     multiPointReSTIR.temporalPosReservoirs[curOffset].shiftPrimaryHitInfo ), true)
//     //             {
//     //                 int2 dis1Vec = abs(selectedPrevBasePixel - prevBasePixel);
//     //                 int dis1 = dis1Vec.x + dis1Vec.y;
//     //                 int2 dis2Vec = abs(curPixel - prevBasePixel);
//     //                 int dis2 = dis2Vec.x + dis2Vec.y;
//     //                 if (dis2 < dis1)
//     //                 {
//     //                     selectedPrevBasePixel = curPixel;
//     //                 }
//     //             }
//     //         }
//     //     }
//     // }
//     // prevBasePixel = selectedPrevBasePixel;
//     // uint prevBufferIndex = getBufferOffset(prevBasePixel, frameDim);
//     // int2 prevShiftPixel = prevBasePixel + shift;
//     // print("motionVector:", motionVector);
//     // print("prevBasePixel", prevBasePixel);
//     // print("curPixel:", basePixel);
//
//     // bool shouldPosReuse = true;
//     // if (multiPointReSTIR.gPathTracer.params.shouldRejectOuterShift &&
//     //    (!multiPointReSTIR.isGeometrySimilar( multiPointReSTIR.posReservoirs[bufferIndex].pathSampleRecord.index == 0 ?
//     //                         multiPointReSTIR.posReservoirs[bufferIndex].pathSampleRecord.pathRcInfo : multiPointReSTIR.posReservoirs[bufferIndex].pathSampleRecord.shiftPathRcInfo,
//     //                         multiPointReSTIR.temporalPosReservoirs[prevBufferIndex].pathSampleRecord.index == 0 ?
//     //                         multiPointReSTIR.temporalPosReservoirs[prevBufferIndex].pathSampleRecord.pathRcInfo : multiPointReSTIR.temporalPosReservoirs[prevBufferIndex].pathSampleRecord.shiftPathRcInfo ) ||
//     //     !multiPointReSTIR.isGeometrySimilar( multiPointReSTIR.posReservoirs[bufferIndex].pathSampleRecord.index == 0 ?
//     //                         multiPointReSTIR.posReservoirs[bufferIndex].pathSampleRecord.shiftPathRcInfo : multiPointReSTIR.posReservoirs[bufferIndex].pathSampleRecord.pathRcInfo,
//     //                         multiPointReSTIR.temporalPosReservoirs[prevBufferIndex].pathSampleRecord.index == 0 ?
//     //                         multiPointReSTIR.temporalPosReservoirs[prevBufferIndex].pathSampleRecord.shiftPathRcInfo : multiPointReSTIR.temporalPosReservoirs[prevBufferIndex].pathSampleRecord.pathRcInfo )))
//     // {
//     //     shouldPosReuse = false;
//     // }
//
//     // bool shouldNegReuse = true;
//     // if (multiPointReSTIR.gPathTracer.params.shouldRejectOuterShift &&
//     //    (!multiPointReSTIR.isGeometrySimilar( multiPointReSTIR.negReservoirs[bufferIndex].pathSampleRecord.index == 0 ?
//     //                         multiPointReSTIR.negReservoirs[bufferIndex].pathSampleRecord.pathRcInfo : multiPointReSTIR.negReservoirs[bufferIndex].pathSampleRecord.shiftPathRcInfo,
//     //                         multiPointReSTIR.temporalNegReservoirs[prevBufferIndex].pathSampleRecord.index == 0 ?
//     //                         multiPointReSTIR.temporalNegReservoirs[prevBufferIndex].pathSampleRecord.pathRcInfo : multiPointReSTIR.temporalNegReservoirs[prevBufferIndex].pathSampleRecord.shiftPathRcInfo ) ||
//     //     !multiPointReSTIR.isGeometrySimilar( multiPointReSTIR.negReservoirs[bufferIndex].pathSampleRecord.index == 0 ?
//     //                         multiPointReSTIR.negReservoirs[bufferIndex].pathSampleRecord.shiftPathRcInfo : multiPointReSTIR.negReservoirs[bufferIndex].pathSampleRecord.pathRcInfo,
//     //                         multiPointReSTIR.temporalNegReservoirs[prevBufferIndex].pathSampleRecord.index == 0 ?
//     //                         multiPointReSTIR.temporalNegReservoirs[prevBufferIndex].pathSampleRecord.shiftPathRcInfo : multiPointReSTIR.temporalNegReservoirs[prevBufferIndex].pathSampleRecord.pathRcInfo )))
//     // {
//     //     shouldNegReuse = false;
//     // }
// #if SHIFT_MAPPING_TYPE == 2
//     bool shouldPosReuse = posFoundNeighbour && isInFrame(prevFracPixel + float(1.0f), frameDim);
//     bool shouldNegReuse = negFoundNeighbour && isInFrame(prevFracPixel + float(1.0f), frameDim);
// #else
//     bool shouldPosReuse = posFoundNeighbour && isInFrame(posPrevBasePixel, frameDim) && isInFrame(posPrevShiftPixel, frameDim);
//     bool shouldNegReuse = negFoundNeighbour && isInFrame(negPrevBasePixel, frameDim) && isInFrame(negPrevShiftPixel, frameDim);
// #endif
//
//
//     if (isInFrame(shiftPixel, frameDim))
//     {
//         PathReservoir posReservoir = multiPointReSTIR.posReservoirs[bufferIndex];
//         PathReservoir negReservoir = multiPointReSTIR.negReservoirs[bufferIndex];
//         posReservoir.reset();
//         negReservoir.reset();
//         RTQueryImpl rtq;
//         {
//             if (shouldPosReuse)
//             {
//                 PathSampleRecord posXi = multiPointReSTIR.posReservoirs[bufferIndex].pathSampleRecord;
//                 PathSampleRecord posXj;
//                 float posJacobianToJ;
// #if SHIFT_MAPPING_TYPE == 2
//                 float2 posXjBaseFracPixelLoc = prevFracPixel;
// #else
//                 float2 posXjBaseFracPixelLoc = float2(posPrevBasePixel) + float2(0.5f);
// #endif
//                 float2 posXjShiftFracPixelLoc = posXjBaseFracPixelLoc + shift;
//                 print("posXjBaseFracPixelLoc:", posXjBaseFracPixelLoc);
//                 float3 posXjIntegrand = multiPointReSTIR.gPathTracer.shiftMappingGPath(posXi,
//                                                                       posXjBaseFracPixelLoc,
//                                                                       multiPointReSTIR.gPathTracer.prevVBuffer[posPrevBufferIndex],
//                                                                       posXjShiftFracPixelLoc,
//                                                                       multiPointReSTIR.gPathTracer.prevVBuffer[posPrevShiftBufferIndex],
//                                                                       rtq, posXj, posJacobianToJ, true);
//                 float3 posYiIntegrand = multiPointReSTIR.posReservoirs[bufferIndex].integrand;
//                 bool isPosXjBaseInScreen = isInFrameFloat(posXjBaseFracPixelLoc + posXi.pathRcInfo.fracPixelLoc, frameDim);
//                 bool isPosXjShiftInScreen = isInFrameFloat(posXjShiftFracPixelLoc + posXi.pathRcInfo.fracPixelLoc, frameDim);
//                 float posDenom = posTargetFunc(posYiIntegrand) +
//                                  (shouldPosReuse ? posTargetFunc(posXjIntegrand) * posTemporalReservoir.C * posJacobianToJ : 0.0f);
//                                  // (shouldNegReuse ? negTargetFunc(posXjIntegrand) * negTemporalReservoir.C * posJacobianToJ : 0.0f);
//                 float posNum = posTargetFunc(posYiIntegrand);
//                 float posMisWeight = (isPosXjBaseInScreen && isPosXjShiftInScreen && posDenom > 0.0f &&
//                                       shouldReuseBasedOnJacobian(posXi.innerJacobian, posXj.innerJacobian)) ? posNum / posDenom : 1.0f;
//
//                 posReservoir.addTemporalSample( posXi, posYiIntegrand, posTargetFunc(posYiIntegrand),
//                     1.0f, posMisWeight, 1.0f,
//                     multiPointReSTIR.posReservoirs[bufferIndex].W,
//                     sampleNext1D(risSg) );
//             }
//
//             if (shouldNegReuse)
//             {
//                 PathSampleRecord negXi = multiPointReSTIR.negReservoirs[bufferIndex].pathSampleRecord;
//                 PathSampleRecord negXj;
//                 float negJacobianToJ;
// #if SHIFT_MAPPING_TYPE == 2
//                 float2 negXjBaseFracPixelLoc = prevFracPixel;
// #else
//                 float2 negXjBaseFracPixelLoc = float2(negPrevBasePixel) + float2(0.5f);
// #endif
//                 float2 negXjShiftFracPixelLoc = negXjBaseFracPixelLoc + shift;
//                 float3 negXjIntegrand = multiPointReSTIR.gPathTracer.shiftMappingGPath(negXi,
//                                                                       negXjBaseFracPixelLoc,
//                                                                       multiPointReSTIR.gPathTracer.prevVBuffer[negPrevBufferIndex],
//                                                                       negXjShiftFracPixelLoc,
//                                                                       multiPointReSTIR.gPathTracer.prevVBuffer[negPrevShiftBufferIndex],
//                                                                       rtq, negXj, negJacobianToJ, true);
//                 float3 negYiIntegrand = multiPointReSTIR.negReservoirs[bufferIndex].integrand;
//                 float negDenom = negTargetFunc(negYiIntegrand) +
//                                  // (shouldPosReuse ? posTargetFunc(negXjIntegrand) * posTemporalReservoir.C * negJacobianToJ : 0.0f) +
//                                  (shouldNegReuse ? negTargetFunc(negXjIntegrand) * negTemporalReservoir.C * negJacobianToJ : 0.0f);
//                 float negNum = negTargetFunc(negYiIntegrand);
//
//                 bool isNegXjBaseInScreen = isInFrameFloat(negXjBaseFracPixelLoc + negXi.pathRcInfo.fracPixelLoc, frameDim);
//                 bool isNegXjShiftInScreen = isInFrameFloat(negXjShiftFracPixelLoc + negXi.pathRcInfo.fracPixelLoc, frameDim);
//                 float negMisWeight = (isNegXjBaseInScreen && isNegXjShiftInScreen && negDenom > 0.0f &&
//                                       shouldReuseBasedOnJacobian(negXi.innerJacobian, negXj.innerJacobian)) ? negNum / negDenom : 1.0f;
//                 negReservoir.addTemporalSample( negXi, negYiIntegrand, negTargetFunc(negYiIntegrand),
//                     1.0f, negMisWeight, 1.0f,
//                     multiPointReSTIR.negReservoirs[bufferIndex].W,
//                     sampleNext1D(risSg) );
//             }
//         }
//
//         {
//             if (shouldPosReuse)
//             {
//                 PathSampleRecord posXi = posTemporalReservoir.pathSampleRecord;
//                 PathSampleRecord posYi;
//                 float posJacobianToCanonical;
// #if SHIFT_MAPPING_TYPE == 2
//                 float2 posYiBaseFracPixelLoc =  float2(basePixel);
// #else
//                 float2 posYiBaseFracPixelLoc =  float2(basePixel) + float2(0.5f);
// #endif
//                 float2 posYiShiftFracPixelLoc =  posYiBaseFracPixelLoc + shift;
//                 float3 posYiIntegrand = multiPointReSTIR.gPathTracer.shiftMappingGPath(posXi,
//                                                                       posYiBaseFracPixelLoc,
//                                                                       multiPointReSTIR.gPathTracer.vbuffer[bufferIndex],
//                                                                       posYiShiftFracPixelLoc,
//                                                                       multiPointReSTIR.gPathTracer.vbuffer[shiftBufferIndex],
//                                                                       rtq,
//                                                                       posYi, posJacobianToCanonical).xyz;
//                 float3 posXiIntegrand = posTemporalReservoir.integrand;
//                 float posDenom = SafeNumber(posTargetFunc(posXiIntegrand) * posTemporalReservoir.C / posJacobianToCanonical) +
//                                  // (shouldNegReuse ? SafeNumber(negTargetFunc(posXiIntegrand) * negTemporalReservoir.C / posJacobianToCanonical) : 0.0f) +
//                                  posTargetFunc(posYiIntegrand);
//                 float posNum = SafeNumber(posTargetFunc(posXiIntegrand) * posTemporalReservoir.C / posJacobianToCanonical);
//                 float posMisWeight = posDenom > 0.0f ? posNum / posDenom : 0.0f;
//                 float posRisSelectU = sampleNext1D(risSg);
//                 if (shouldReuseBasedOnJacobian(posXi.innerJacobian, posYi.innerJacobian))
//                 {
//                     posReservoir.addTemporalSample( posYi,
//                         posYiIntegrand,
//                         posTargetFunc(posYiIntegrand),
//                         posJacobianToCanonical,
//                         posMisWeight,
//                         posTemporalReservoir.C,
//                         posTemporalReservoir.W,
//                         posRisSelectU );
//                 }
//             }
//
//             if (shouldNegReuse)
//             {
//                 PathSampleRecord negXi = negTemporalReservoir.pathSampleRecord;
//                 PathSampleRecord negYi;
//                 float negJacobianToCanonical;
// #if SHIFT_MAPPING_TYPE == 2
//                 float2 negYiBaseFracPixelLoc = float2(basePixel);
// #else
//                 float2 negYiBaseFracPixelLoc = float2(basePixel) + float2(0.5f);
// #endif
//                 float2 negYiShiftFracPixelLoc = negYiBaseFracPixelLoc + shift;
//                 float3 negYiIntegrand = multiPointReSTIR.gPathTracer.shiftMappingGPath(negXi,
//                                                                       negYiBaseFracPixelLoc,
//                                                                       multiPointReSTIR.gPathTracer.vbuffer[bufferIndex],
//                                                                       negYiShiftFracPixelLoc,
//                                                                       multiPointReSTIR.gPathTracer.vbuffer[shiftBufferIndex],
//                                                                       rtq,
//                                                                       negYi, negJacobianToCanonical).xyz;
//                 float3 negXiIntegrand = negTemporalReservoir.integrand;
//                 float negDenom = SafeNumber(negTargetFunc(negXiIntegrand) * negTemporalReservoir.C / negJacobianToCanonical) +
//                                  // (shouldPosReuse ? SafeNumber(posTargetFunc(negXiIntegrand) * posTemporalReservoir.C / negJacobianToCanonical) : 0.0f) +
//                                  negTargetFunc(negYiIntegrand);
//                 float negNum = SafeNumber(negTargetFunc(negXiIntegrand) * negTemporalReservoir.C / negJacobianToCanonical);
//                 float negMisWeight = negDenom > 0.0f ? negNum / negDenom : 0.0f;
//                 float negRisSelectU = sampleNext1D(risSg);
//                 if (shouldReuseBasedOnJacobian(negXi.innerJacobian, negYi.innerJacobian))
//                 {
//                     negReservoir.addTemporalSample( negYi,
//                         negYiIntegrand,
//                         negTargetFunc(negYiIntegrand),
//                         negJacobianToCanonical,
//                         negMisWeight,
//                         negTemporalReservoir.C,
//                         negTemporalReservoir.W,
//                         negRisSelectU );
//                 }
//             }
//         }
//         posReservoir.finalizeTemporalReuse(multiPointReSTIR.gPathTracer.params.historyLength);
//         negReservoir.finalizeTemporalReuse(multiPointReSTIR.gPathTracer.params.historyLength);
//         if (shouldPosReuse)
//         {
//             multiPointReSTIR.posTmpReservoirs[bufferIndex] = posReservoir;
//         }
//         if (shouldNegReuse)
//         {
//             multiPointReSTIR.negTmpReservoirs[bufferIndex] = negReservoir;
//         }
//     }
}

void MultiPointSpatialReuse(MultiPointReSTIR multiPointReSTIR, uint2 basePixel, uint2 shift)
{
//     uint bufferIndex = getBufferOffset(basePixel, multiPointReSTIR.gPathTracer.params.frameDim);
//
//     int2 shiftPixel = basePixel + shift;
//     uint shiftBufferIndex = getBufferOffset(shiftPixel, multiPointReSTIR.gPathTracer.params.frameDim);
//     SampleGenerator risSg = SampleGenerator(basePixel,
//                                     multiPointReSTIR.gPathTracer.params.seed * (kNumInitialSamples + 4));
//     uint startIndex = uint(sampleNext1D(risSg) * kNeighborOffsetCount);
//     if (isInFrame(shiftPixel, multiPointReSTIR.gPathTracer.params.frameDim))
//     {
//         uint posNeighboursValidMask;
//         uint negNeighboursValidMask;
//         float posCTot = 1.0f;
//         float negCTot = 1.0f;
//         for (int i = 0; i < kNumSpatialNeighbours; ++i)
//         {
//             int2 curXiPixel = multiPointReSTIR.getNextNeighborPixel(startIndex, basePixel, i);
//
//             if (!isInFrame(curXiPixel, multiPointReSTIR.gPathTracer.params.frameDim) ||
//                 !isInFrame(curXiPixel + shift, multiPointReSTIR.gPathTracer.params.frameDim))
//                 continue;
//
//             uint curBaseOffset = getBufferOffset(curXiPixel, multiPointReSTIR.gPathTracer.params.frameDim);
//             if (!multiPointReSTIR.spatialPosReservoirs[curBaseOffset].pathSampleRecord.isNullSample)
//             {
//                 posNeighboursValidMask = setBit(posNeighboursValidMask, i);
//             }
//             if (!multiPointReSTIR.spatialNegReservoirs[curBaseOffset].pathSampleRecord.isNullSample)
//             {
//                 negNeighboursValidMask = setBit(negNeighboursValidMask, i);
//             }
//
//             // if (multiPointReSTIR.gPathTracer.params.shouldRejectOuterShift &&
//             //    (!multiPointReSTIR.isGeometrySimilar( multiPointReSTIR.spatialPosReservoirs[bufferIndex].pathSampleRecord.index == 0 ?
//             //                         multiPointReSTIR.spatialPosReservoirs[bufferIndex].pathSampleRecord.pathRcInfo : multiPointReSTIR.spatialPosReservoirs[bufferIndex].pathSampleRecord.shiftPathRcInfo,
//             //                         multiPointReSTIR.spatialPosReservoirs[curBaseOffset].pathSampleRecord.index == 0 ?
//             //                         multiPointReSTIR.spatialPosReservoirs[curBaseOffset].pathSampleRecord.pathRcInfo : multiPointReSTIR.spatialPosReservoirs[curBaseOffset].pathSampleRecord.shiftPathRcInfo ) ||
//             //     !multiPointReSTIR.isGeometrySimilar( multiPointReSTIR.spatialPosReservoirs[bufferIndex].pathSampleRecord.index == 0 ?
//             //                         multiPointReSTIR.spatialPosReservoirs[bufferIndex].pathSampleRecord.shiftPathRcInfo : multiPointReSTIR.spatialPosReservoirs[bufferIndex].pathSampleRecord.pathRcInfo,
//             //                         multiPointReSTIR.spatialPosReservoirs[curBaseOffset].pathSampleRecord.index == 0 ?
//             //                         multiPointReSTIR.spatialPosReservoirs[curBaseOffset].pathSampleRecord.shiftPathRcInfo : multiPointReSTIR.spatialPosReservoirs[curBaseOffset].pathSampleRecord.pathRcInfo )))
//             // {
//             //     posNeighboursValidMask = clearBit(posNeighboursValidMask, i);
//             // }
//
//             // if (multiPointReSTIR.gPathTracer.params.shouldRejectOuterShift &&
//             //    (!multiPointReSTIR.isGeometrySimilar( multiPointReSTIR.spatialNegReservoirs[bufferIndex].pathSampleRecord.index == 0 ?
//             //                         multiPointReSTIR.spatialNegReservoirs[bufferIndex].pathSampleRecord.pathRcInfo : multiPointReSTIR.spatialNegReservoirs[bufferIndex].pathSampleRecord.shiftPathRcInfo,
//             //                         multiPointReSTIR.spatialNegReservoirs[curBaseOffset].pathSampleRecord.index == 0 ?
//             //                         multiPointReSTIR.spatialNegReservoirs[curBaseOffset].pathSampleRecord.pathRcInfo : multiPointReSTIR.spatialNegReservoirs[curBaseOffset].pathSampleRecord.shiftPathRcInfo ) ||
//             //     !multiPointReSTIR.isGeometrySimilar( multiPointReSTIR.spatialNegReservoirs[bufferIndex].pathSampleRecord.index == 0 ?
//             //                         multiPointReSTIR.spatialNegReservoirs[bufferIndex].pathSampleRecord.shiftPathRcInfo : multiPointReSTIR.spatialNegReservoirs[bufferIndex].pathSampleRecord.pathRcInfo,
//             //                         multiPointReSTIR.spatialNegReservoirs[curBaseOffset].pathSampleRecord.index == 0 ?
//             //                         multiPointReSTIR.spatialNegReservoirs[curBaseOffset].pathSampleRecord.shiftPathRcInfo : multiPointReSTIR.spatialNegReservoirs[curBaseOffset].pathSampleRecord.pathRcInfo )))
//             // {
//             //     negNeighboursValidMask = clearBit(negNeighboursValidMask, i);
//             // }
//         }
//
//         for (int j = 0; j < kNumSpatialNeighbours; ++j)
//         {
//             int2 curXjPixel = multiPointReSTIR.getNextNeighborPixel(startIndex, basePixel, j);
//             if (multiPointReSTIR.shouldSpatialReuse(j, posNeighboursValidMask))
//             {
//                 posCTot += 1.0f;
//             }
//
//             if (multiPointReSTIR.shouldSpatialReuse(j, negNeighboursValidMask))
//             {
//                 negCTot += 1.0f;
//             }
//         }
//
//         RTQueryImpl rtq;
//         PathReservoir posReservoir = multiPointReSTIR.spatialPosReservoirs[bufferIndex];
//         PathReservoir negReservoir = multiPointReSTIR.spatialNegReservoirs[bufferIndex];
//         posReservoir.reset();
//         negReservoir.reset();
//         for (int i = -1; i < kNumSpatialNeighbours; ++i)
//         {
//             int2 curXiPixel;
//             if (i == -1)
//             {
//                 // Get the spatial neighbor pixel
//                 curXiPixel = basePixel;
//             }
//             else
//             {
//                 curXiPixel = multiPointReSTIR.getNextNeighborPixel(startIndex, basePixel, i);
//             }
//
//             if (!multiPointReSTIR.shouldSpatialReuse(i,
//                     posNeighboursValidMask |
//                     negNeighboursValidMask))
//                 continue;
//
//             if (!isInFrame(curXiPixel, multiPointReSTIR.gPathTracer.params.frameDim) ||
//                 !isInFrame(curXiPixel + shift, multiPointReSTIR.gPathTracer.params.frameDim))
//                 continue;
//
//             float posCi = multiPointReSTIR.shouldSpatialReuse(i, posNeighboursValidMask) ? 1.0f : 0.0f;
//             float negCi = multiPointReSTIR.shouldSpatialReuse(i, posNeighboursValidMask) ? 1.0f : 0.0f;
//
//             uint curBaseOffset = getBufferOffset(curXiPixel, multiPointReSTIR.gPathTracer.params.frameDim);
//
//             PathSampleRecord posXi = multiPointReSTIR.spatialPosReservoirs[curBaseOffset].pathSampleRecord;
//             float3 posXiIntegrand = multiPointReSTIR.spatialPosReservoirs[curBaseOffset].integrand;
//             PathSampleRecord posYi;
//             float3 posYiIntegrand;
//             float posJacobianToCanonical;
//             if (i == -1)
//             {
//                 posYi = posXi;
//                 posJacobianToCanonical = 1.0f;
//                 posYiIntegrand = posXiIntegrand;
//             }
//             else
//             {
//                 float2 posBaseFracPixelLoc  = basePixel;
//                 float2 posShiftFracPixelLoc = posBaseFracPixelLoc + shift;
//                 posYiIntegrand = multiPointReSTIR.gPathTracer.shiftMappingGPath( posXi,
//                     posBaseFracPixelLoc,
//                     multiPointReSTIR.gPathTracer.vbuffer[bufferIndex],
//                     posShiftFracPixelLoc,
//                     multiPointReSTIR.gPathTracer.vbuffer[shiftBufferIndex],
//                     rtq,
//                     posYi, posJacobianToCanonical );
//                 posJacobianToCanonical = shouldReuseBasedOnJacobian(posXi.innerJacobian, posYi.innerJacobian) ? posJacobianToCanonical : 0.0f;
//             }
//
//             PathSampleRecord negXi = multiPointReSTIR.spatialNegReservoirs[curBaseOffset].pathSampleRecord;
//             float3 negXiIntegrand = multiPointReSTIR.spatialNegReservoirs[curBaseOffset].integrand;
//             float3 negYiIntegrand;
//             PathSampleRecord negYi;
//             float negJacobianToCanonical;
//             if (i == -1)
//             {
//                 negYi = negXi;
//                 negJacobianToCanonical = 1.0f;
//                 negYiIntegrand = negXiIntegrand;
//             }
//             else
//             {
//                 float2 negBaseFracPixelLoc =  basePixel;
//                 float2 negShiftFracPixelLoc = negBaseFracPixelLoc + shift;
//                 negXiIntegrand = multiPointReSTIR.spatialNegReservoirs[curBaseOffset].integrand;
//                 negYiIntegrand = multiPointReSTIR.gPathTracer.shiftMappingGPath( negXi,
//                     negBaseFracPixelLoc,
//                     multiPointReSTIR.gPathTracer.vbuffer[bufferIndex],
//                     negShiftFracPixelLoc,
//                     multiPointReSTIR.gPathTracer.vbuffer[shiftBufferIndex],
//                     rtq,
//                     negYi, negJacobianToCanonical );
//                 negJacobianToCanonical = shouldReuseBasedOnJacobian(negXi.innerJacobian, negYi.innerJacobian) ? negJacobianToCanonical : 0.0f;
//             }
//
//             float posPosMisWeight = 0.0f;
//             float posNegMisWeight = 0.0f;
//             float negNegMisWeight = 0.0f;
//             float negPosMisWeight = 0.0f;
//             float posCC = 1.0f;
//             float negCC = 1.0f;
//             if (i == -1)
//             {
//                 float mPosNum = posCC * posTargetFunc(posYiIntegrand);
//                 float mNegNum = negCC * negTargetFunc(negYiIntegrand);
//
//                 posPosMisWeight += posCC / posCTot;
//                 negNegMisWeight += negCC / negCTot;
//
//                 for (int j = 0; j < kNumSpatialNeighbours; ++j)
//                 {
//                     int2 curXjPixel = multiPointReSTIR.getNextNeighborPixel(startIndex, basePixel, j);
//
//                     if (!multiPointReSTIR.shouldSpatialReuse(j,
//                             posNeighboursValidMask |
//                             negNeighboursValidMask))
//                         continue;
//
//                     uint shiftOffset = getBufferOffset(curXjPixel, multiPointReSTIR.gPathTracer.params.frameDim);
//                     uint shiftShiftOffset = getBufferOffset(curXjPixel + shift, multiPointReSTIR.gPathTracer.params.frameDim);
//                     float posCj = multiPointReSTIR.shouldSpatialReuse(j, posNeighboursValidMask) ? 1.0f : 0.0f;
//                     float negCj = multiPointReSTIR.shouldSpatialReuse(j, negNeighboursValidMask) ? 1.0f : 0.0f;
//
//                     if (multiPointReSTIR.shouldSpatialReuse(j, posNeighboursValidMask))
//                     {
//                         PathSampleRecord posXj;
//                         float jacobianToJ;
//                         float2 posBaseFracPixelLoc = curXjPixel;
//                         float2 posShiftFracPixelLoc = posBaseFracPixelLoc + shift;
//                         float3 posXjIntegrand = multiPointReSTIR.gPathTracer.shiftMappingGPath(posYi,
//                                                                               posBaseFracPixelLoc,
//                                                                               multiPointReSTIR.gPathTracer.vbuffer[shiftOffset],
//                                                                               posShiftFracPixelLoc,
//                                                                               multiPointReSTIR.gPathTracer.vbuffer[shiftShiftOffset],
//                                                                               rtq,
//                                                                               posXj, jacobianToJ);
//                         if (shouldReuseBasedOnJacobian(posYi.innerJacobian, posXj.innerJacobian))
//                         {
//                             float mPosDen = mPosNum + (posCTot - posCC) * posTargetFunc(posXjIntegrand) * jacobianToJ;
//                             float mPosW = posCj / posCTot;
//                             posPosMisWeight += mPosDen > 0.0f ? mPosW * mPosNum / mPosDen : 0.0f;
//                         }
//                         else
//                         {
//                             posPosMisWeight += posCj / posCTot;
//                         }
//                     }
//
//                     if (multiPointReSTIR.shouldSpatialReuse(j, negNeighboursValidMask))
//                     {
//                         PathSampleRecord negXj;
//                         float jacobianToJ;
//                         float2 negBaseFracPixelLoc  = curXjPixel;
//                         float2 negShiftFracPixelLoc = negBaseFracPixelLoc + shift;
//                         float3 negXjIntegrand = multiPointReSTIR.gPathTracer.shiftMappingGPath(negYi,
//                                                                               negBaseFracPixelLoc,
//                                                                               multiPointReSTIR.gPathTracer.vbuffer[shiftOffset],
//                                                                               negShiftFracPixelLoc,
//                                                                               multiPointReSTIR.gPathTracer.vbuffer[shiftShiftOffset],
//                                                                               rtq,
//                                                                               negXj, jacobianToJ);
//                         if (shouldReuseBasedOnJacobian(negXj.innerJacobian, negYi.innerJacobian))
//                         {
//                             float mNegDen = mNegNum + (negCTot - negCC) * negTargetFunc(negXjIntegrand) * jacobianToJ;
//                             float mNegW = negCj / negCTot;
//                             negNegMisWeight += mNegDen > 0.0f ? mNegW * mNegNum / mNegDen : 0.0f;
//                         }
//                         else
//                         {
//                             negNegMisWeight += negCj / negCTot;
//                         }
//                     }
//                 }
//             }
//             else
//             {
//                 if (multiPointReSTIR.shouldSpatialReuse(i, posNeighboursValidMask) &&
//                     shouldReuseBasedOnJacobian(posXi.innerJacobian, posYi.innerJacobian))
//                 {
//                     float posPHatToI = posJacobianToCanonical > 0.0f ? posTargetFunc(posXiIntegrand) / posJacobianToCanonical : 0.0f;
//                     float mPosNum = posCi * posPHatToI;
//                     float mPosDen = (posCTot - posCC) * posPHatToI +
//                                     posCC * posTargetFunc(posYiIntegrand);
//                     float posW = (posCTot - posCC) / posCTot;
//                     posPosMisWeight = mPosDen > 0.0f ? posW * mPosNum / mPosDen : 0.0f;
//                 }
//
//                 if (multiPointReSTIR.shouldSpatialReuse(i, negNeighboursValidMask) &&
//                     shouldReuseBasedOnJacobian(negXi.innerJacobian, negYi.innerJacobian))
//                 {
//                     float negPHatToI = negJacobianToCanonical > 0.0f ? negTargetFunc(negXiIntegrand) / negJacobianToCanonical : 0.0f;
//                     float mNegNum = negCi * negPHatToI;
//                     float mNegDen = (negCTot - negCC) * negPHatToI +
//                                     negCC * negTargetFunc(negYiIntegrand);
//                     float negW = (negCTot - negCC) / negCTot;
//                     negNegMisWeight = mNegDen > 0.0f ? negW * mNegNum / mNegDen : 0.0f;
//                 }
//             }
//
//             if (multiPointReSTIR.shouldSpatialReuse(i, posNeighboursValidMask))
//             {
//                 float posRisSelectU = sampleNext1D(risSg);
//                 posReservoir.addSpatialSample( posYi,
//                     posYiIntegrand,
//                     posTargetFunc(posYiIntegrand),
//                     posJacobianToCanonical,
//                     posPosMisWeight,
//                     multiPointReSTIR.spatialPosReservoirs[curBaseOffset].C,
//                     multiPointReSTIR.spatialPosReservoirs[curBaseOffset].W,
//                     posRisSelectU );
//             }
//
//             if (multiPointReSTIR.shouldSpatialReuse(i, negNeighboursValidMask))
//             {
//                 float negRisSelectU = sampleNext1D(risSg);
//                 negReservoir.addSpatialSample( negYi,
//                     negYiIntegrand,
//                     negTargetFunc(negYiIntegrand),
//                     negJacobianToCanonical,
//                     negNegMisWeight,
//                     multiPointReSTIR.spatialNegReservoirs[curBaseOffset].C,
//                     multiPointReSTIR.spatialNegReservoirs[curBaseOffset].W,
//                     negRisSelectU );
//             }
//         }
//
//         posReservoir.finalizeSpatialReuse(multiPointReSTIR.gPathTracer.params.historyLength);
//         negReservoir.finalizeSpatialReuse(multiPointReSTIR.gPathTracer.params.historyLength);
//         multiPointReSTIR.posReservoirs[bufferIndex] = posReservoir;
//         multiPointReSTIR.negReservoirs[bufferIndex] = negReservoir;
//     }
}
