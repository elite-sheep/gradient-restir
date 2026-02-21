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

MultiPointSpatialReuse gGradientIntegrator;
ParameterBlock<GPathTracer> gPathTracer;
ParameterBlock<MultiPointReSTIR> multiPointReSTIR;
uniform uint shiftDir;
uniform uint pathsPerPixel;
RWTexture2D<float4> shouldSpatialReuse;
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

struct MultiPointSpatialReuse
{
    void run(uint2 basePixel)
    {
        uint2 frameDim = multiPointReSTIR.gPathTracer.params.frameDim;
        uint basePixelBufferIndex = getBufferOffset(basePixel, frameDim);
        uint2 shift = (shiftDir == 0) ? uint2(1, 0) : uint2(0, 1);
        uint2 shiftPixel = basePixel + shift;
        uint finalPathBufferIndex = getBufferOffset(basePixel, frameDim, pathsPerPixel) + pathsPerPixel - 1;
        SampleGenerator risSg = SampleGenerator(basePixel,
                                        multiPointReSTIR.gPathTracer.params.seed * (kNumInitialSamples + 3));
        uint startIndex = uint(sampleNext1D(risSg) * kNeighborOffsetCount);

        int2 neighbourPixel = neighbourPixels[basePixel];
        bool shouldReuse = !all(basePixel == neighbourPixel) && kNumSpatialNeighbours > 0;
        if (!shouldReuse)
        {
            uint curBaseOffset = getBufferOffset(basePixel, frameDim, pathsPerPixel);
            multiPointReSTIR.posReservoirs[finalPathBufferIndex] = multiPointReSTIR.posReservoirs[curBaseOffset];
            multiPointReSTIR.posPaths[finalPathBufferIndex] = multiPointReSTIR.posPaths[curBaseOffset];
            multiPointReSTIR.posReservoirs[finalPathBufferIndex].pathID = finalPathBufferIndex;
            // uint curBaseOffset = getBufferOffset(basePixel, frameDim, pathsPerPixel);
            // multiPointReSTIR.posReservoirs[finalPathBufferIndex] = multiPointReSTIR.posReservoirs[curBaseOffset];
            return;
        }
        shouldSpatialReuse[basePixel] = shouldReuse ? float4(1.0f, 1.0f, 1.0f, 1.0f) : float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (isInFrame(basePixel, multiPointReSTIR.gPathTracer.params.frameDim) &&
            isInFrame(shiftPixel, multiPointReSTIR.gPathTracer.params.frameDim))
        {
            uint posNeighboursValidMask = 0;
            float posCTot = 1.0f;
            for (int i = 0; i < kNumSpatialNeighbours; ++i)
            {
                int2 curXiPixel = neighbourPixel;
                uint curXiOffset = getBufferOffset(curXiPixel, multiPointReSTIR.gPathTracer.params.frameDim, pathsPerPixel);

                if (!isInFrame(curXiPixel, multiPointReSTIR.gPathTracer.params.frameDim) ||
                    !isInFrame(curXiPixel + shift, multiPointReSTIR.gPathTracer.params.frameDim))
                    continue;

                uint curBaseOffset = getBufferOffset(curXiPixel, frameDim, pathsPerPixel);
                if (!multiPointReSTIR.posReservoirs[curBaseOffset].isNullSample)
                    posNeighboursValidMask = setBit(posNeighboursValidMask, i);
            }
            // posNeighboursValidMask = 0;

            for (int j = 0; j < kNumSpatialNeighbours; ++j)
            {
                int2 curXjPixel = neighbourPixel;
                if (multiPointReSTIR.shouldSpatialReuse(j, posNeighboursValidMask))
                {
                    posCTot += 1.0f;
                }
            }

            RTQueryImpl rtq;
            PathReservoir posReservoir; posReservoir.init();
            {
                uint curBaseOffset = getBufferOffset(basePixel, frameDim, pathsPerPixel);
                {
                    PathSampleRecord posPath;
                    PathReservoir posXiReservoir = multiPointReSTIR.posReservoirs[curBaseOffset];
                    PathSampleRecord posYi = multiPointReSTIR.posPaths[posXiReservoir.pathID];
                    float3 posYiIntegrand = posXiReservoir.integrand;
                    float posMisWeight = 0.0f;
                    float posCC = 1.0f;
                    float mPosNum = posCC * posTargetFunc(posYiIntegrand);
                    posMisWeight += posCC / posCTot;
                    for (int j = 0; j < kNumSpatialNeighbours; ++j)
                    {
                        int2 curXjPixel = neighbourPixel;

                        if (!multiPointReSTIR.shouldSpatialReuse(j, posNeighboursValidMask))
                            continue;

                        float posCj = multiPointReSTIR.shouldSpatialReuse(j, posNeighboursValidMask) ? 1.0f : 0.0f;
                        if (multiPointReSTIR.shouldSpatialReuse(j, posNeighboursValidMask))
                        {
                            PathSampleRecord posXj;
                            float jacobianToJ;
                            float2 posBaseFracPixelLoc = curXjPixel;
                            float2 posShiftFracPixelLoc = posBaseFracPixelLoc + shift;
                            float3 posXjBaseIntegrand = gPathTracer.areaReSTIRShiftMapping(posYi,
                                                    (posYi.index == 0) ? posBaseFracPixelLoc : posShiftFracPixelLoc,
                                                    false, rtq, posXj,
                                                    jacobianToJ).xyz;
                            posXj.index = posYi.index;
                            float3 posXjIntegrand = gPathTracer.shiftAndComputeGPath(posXj,
                                                                          posBaseFracPixelLoc,
                                                                          posXjBaseIntegrand,
                                                                          posShiftFracPixelLoc,
                                                                          rtq,
                                                                          false);
                            if (shouldReuseBasedOnJacobian(posXj.innerJacobian, posYi.innerJacobian))
                            {
                                float posPHatToJ = posTargetFunc(posXjIntegrand) * jacobianToJ;
                                float mPosNum = (posCTot - posCC) * posPHatToJ;
                                float mPosDen = mPosNum +
                                                posCC * posTargetFunc(posYiIntegrand);
                                float mPosW = posCj / posCTot;
                                posMisWeight += mPosW - (mPosDen > 0.0f ? mPosW * mPosNum / mPosDen : 0.0f);
                            }
                            else
                            {
                                posMisWeight += posCj / posCTot;
                            }
                        }
                    }

                    if (!posXiReservoir.isNullSample)
                    {
                        float posRisSelectU = sampleNext1D(risSg);
                        bool isSelected = posReservoir.addSpatialSample( finalPathBufferIndex,
                                                                         posYiIntegrand,
                                                                         posTargetFunc(posYiIntegrand),
                                                                         1.0f,
                                                                         posMisWeight,
                                                                         posXiReservoir.C,
                                                                         posXiReservoir.W,
                                                                         posRisSelectU );
                        posPath = isSelected ? posYi : posPath;
                    }

                    for (int i = 0; i < kNumSpatialNeighbours; ++i)
                    {
                        int2 curXiPixel = neighbourPixel;

                        if (!multiPointReSTIR.shouldSpatialReuse(i, posNeighboursValidMask))
                            continue;

                        float posCi = multiPointReSTIR.shouldSpatialReuse(i, posNeighboursValidMask) ? 1.0f : 0.0f;
                        uint curBaseOffset = getBufferOffset(curXiPixel, frameDim, pathsPerPixel);

                        PathReservoir posXiReservoir = multiPointReSTIR.posReservoirs[curBaseOffset];
                        PathSampleRecord posXi = multiPointReSTIR.posPaths[posXiReservoir.pathID];
                        float3 posXiIntegrand = posXiReservoir.integrand;
                        PathSampleRecord posYi;
                        float3 posYiIntegrand;
                        float posJacobianToCanonical;
                        if (multiPointReSTIR.shouldSpatialReuse(i, posNeighboursValidMask))
                        {
                            float2 posYiBasePixel = posXi.index == 0 ? basePixel : shiftPixel;
                            float3 posYiBaseIntegrand = gPathTracer.areaReSTIRShiftMapping(posXi,
                                                        posYiBasePixel,
                                                        false, rtq, posYi,
                                                        posJacobianToCanonical).xyz;
                            posYi.index = posXi.index;
                            posYiIntegrand = gPathTracer.shiftAndComputeGPath(posYi,
                                                                              basePixel,
                                                                              posYiBaseIntegrand,
                                                                              shiftPixel,
                                                                              rtq,
                                                                              false);
                        }

                        float posMisWeight = 0.0f;
                        float posCC = 1.0f;
                        {
                            if (multiPointReSTIR.shouldSpatialReuse(i, posNeighboursValidMask) &&
                                shouldReuseBasedOnJacobian(posXi.innerJacobian, posYi.innerJacobian))
                            {
                                float posPHatToI = posJacobianToCanonical > 0.0f ? posTargetFunc(posXiIntegrand) / posJacobianToCanonical : 0.0f;
                                float mPosNum = (posCTot - posCC) * posPHatToI;
                                float mPosDen = mPosNum +
                                                posCC * posTargetFunc(posYiIntegrand);
                                float posW = posCi / posCTot;
                                posMisWeight = mPosDen > 0.0f ? posW * mPosNum / mPosDen : 0.0f;
                            }
                        }

                        if (multiPointReSTIR.shouldSpatialReuse(i, posNeighboursValidMask))
                        {
                            if (posXiReservoir.W > 1e6)
                            {
                                print("fracPixelLoc:", posXi.pathRcInfo.fracPixelLoc);
                                print("isLastVertexNEE:", posXi.pathRcInfo.isLastVertexNEE);
                                print("pathLength:", posXi.pathRcInfo.pathLength);
                                print("isDeltaOnlyPath:", posXi.pathRcInfo.isDeltaOnlyPath);
                                print("rcIndex:", posXi.pathRcInfo.rcHitIndex);
                                print("curYiIntegrand:", posYiIntegrand);
                                print("curXiIntegrand:", posXiIntegrand);
                            }
                            float posRisSelectU = sampleNext1D(risSg);
                            bool isSelected = posReservoir.addSpatialSample( finalPathBufferIndex,
                                                                             posYiIntegrand,
                                                                             posTargetFunc(posYiIntegrand),
                                                                             posJacobianToCanonical,
                                                                             posMisWeight,
                                                                             posXiReservoir.C,
                                                                             posXiReservoir.W,
                                                                             posRisSelectU );
                            posPath = isSelected ? posYi : posPath;
                        }
                    }
                    multiPointReSTIR.posPaths[finalPathBufferIndex] = posPath;
                }
            }

            posReservoir.finalizeSpatialReuse(multiPointReSTIR.gPathTracer.params.historyLength);
            multiPointReSTIR.posReservoirs[finalPathBufferIndex] = posReservoir;
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
