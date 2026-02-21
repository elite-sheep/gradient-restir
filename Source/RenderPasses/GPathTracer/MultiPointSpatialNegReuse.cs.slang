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
                                        multiPointReSTIR.gPathTracer.params.seed * (kNumInitialSamples + 4));
        uint startIndex = uint(sampleNext1D(risSg) * kNeighborOffsetCount);

        int2 neighbourPixel = neighbourPixels[basePixel];
        bool shouldReuse = kNumSpatialNeighbours > 0 && !all(basePixel == neighbourPixel);
        if (!shouldReuse)
        {
            uint curBaseOffset = getBufferOffset(basePixel, frameDim, pathsPerPixel);
            multiPointReSTIR.negReservoirs[finalPathBufferIndex] = multiPointReSTIR.negReservoirs[curBaseOffset];
            multiPointReSTIR.negPaths[finalPathBufferIndex] = multiPointReSTIR.negPaths[curBaseOffset];
            multiPointReSTIR.negReservoirs[finalPathBufferIndex].pathID = finalPathBufferIndex;
            return;
        }

        if (isInFrame(basePixel, multiPointReSTIR.gPathTracer.params.frameDim) &&
            isInFrame(shiftPixel, multiPointReSTIR.gPathTracer.params.frameDim))
        {
            uint negNeighboursValidMask = 0;
            float negCTot = 1.0f;
            for (int i = 0; i < kNumSpatialNeighbours; ++i)
            {
                int2 curXiPixel = neighbourPixel;
                uint curXiOffset = getBufferOffset(curXiPixel, multiPointReSTIR.gPathTracer.params.frameDim, pathsPerPixel);

                if (!isInFrame(curXiPixel, multiPointReSTIR.gPathTracer.params.frameDim) ||
                    !isInFrame(curXiPixel + shift, multiPointReSTIR.gPathTracer.params.frameDim))
                    continue;

                uint curBaseOffset = getBufferOffset(curXiPixel, frameDim, pathsPerPixel);
                if (!multiPointReSTIR.negReservoirs[curBaseOffset].isNullSample)
                    negNeighboursValidMask = setBit(negNeighboursValidMask, i);
            }
            // posNeighboursValidMask = 0;
            // negNeighboursValidMask = 0;

            for (int j = 0; j < kNumSpatialNeighbours; ++j)
            {
                int2 curXjPixel = neighbourPixel;

                if (multiPointReSTIR.shouldSpatialReuse(j, negNeighboursValidMask))
                {
                    negCTot += 1.0f;
                }
            }

            RTQueryImpl rtq;
            PathReservoir negReservoir; negReservoir.init();
            {
                uint curBaseOffset = getBufferOffset(basePixel, frameDim, pathsPerPixel);
                {
                    PathSampleRecord negPath;
                    PathReservoir negXiReservoir = multiPointReSTIR.negReservoirs[curBaseOffset];
                    PathSampleRecord negYi = multiPointReSTIR.negPaths[negXiReservoir.pathID];
                    float3 negYiIntegrand = negXiReservoir.integrand;
                    float negMisWeight = 0.0f;
                    float negCC = 1.0f;
                    float mNegNum = negCC * negTargetFunc(negYiIntegrand);
                    negMisWeight += negCC / negCTot;

                    for (int j = 0; j < kNumSpatialNeighbours; ++j)
                    {
                        int2 curXjPixel = neighbourPixel;

                        if (!multiPointReSTIR.shouldSpatialReuse(j, negNeighboursValidMask))
                            continue;

                        float negCj = multiPointReSTIR.shouldSpatialReuse(j, negNeighboursValidMask) ? 1.0f : 0.0f;
                        if (multiPointReSTIR.shouldSpatialReuse(j, negNeighboursValidMask))
                        {
                            PathSampleRecord negXj;
                            float jacobianToJ;
                            float2 negBaseFracPixelLoc  = curXjPixel;
                            float2 negShiftFracPixelLoc = negBaseFracPixelLoc + shift;
                            float3 negXjBaseIntegrand = multiPointReSTIR.gPathTracer.areaReSTIRShiftMapping(negYi,
                                                    (negYi.index == 0) ? negBaseFracPixelLoc : negShiftFracPixelLoc,
                                                    false, rtq, negXj,
                                                    jacobianToJ).xyz;
                            negXj.index = negYi.index;
                            float3 negXjIntegrand = multiPointReSTIR.gPathTracer.shiftAndComputeGPath(negXj,
                                                                                        negBaseFracPixelLoc,
                                                                                        negXjBaseIntegrand,
                                                                                        negShiftFracPixelLoc,
                                                                                        rtq,
                                                                                        false);
                            if (shouldReuseBasedOnJacobian(negXj.innerJacobian, negYi.innerJacobian))
                            {
                                float mNegDen = mNegNum + (negCTot - negCC) * negTargetFunc(negXjIntegrand) * jacobianToJ;
                                float mNegW = negCj / negCTot;
                                negMisWeight += mNegDen > 0.0f ? mNegW * mNegNum / mNegDen : 0.0f;
                            }
                            else
                            {
                                negMisWeight += negCj / negCTot;
                            }
                        }
                    }

                    if (!negXiReservoir.isNullSample)
                    {
                        float negRisSelectU = sampleNext1D(risSg);
                        bool isSelected = negReservoir.addSpatialSample( finalPathBufferIndex,
                                                                         negYiIntegrand,
                                                                         negTargetFunc(negYiIntegrand),
                                                                         1.0f,
                                                                         negMisWeight,
                                                                         negXiReservoir.C,
                                                                         negXiReservoir.W,
                                                                         negRisSelectU );
                        negPath = isSelected ? negYi : negPath;
                    }

                    for (int i = 0; i < kNumSpatialNeighbours; ++i)
                    {
                        int2 curXiPixel = neighbourPixel;

                        if (!multiPointReSTIR.shouldSpatialReuse(i, negNeighboursValidMask))
                            continue;

                        float negCi = multiPointReSTIR.shouldSpatialReuse(i, negNeighboursValidMask) ? 1.0f : 0.0f;

                        uint curBaseOffset = getBufferOffset(curXiPixel, frameDim, pathsPerPixel);
                        PathReservoir negXiReservoir = multiPointReSTIR.negReservoirs[curBaseOffset];
                        PathSampleRecord negXi = multiPointReSTIR.negPaths[negXiReservoir.pathID];
                        float3 negXiIntegrand = negXiReservoir.integrand;
                        float3 negYiIntegrand;
                        PathSampleRecord negYi;
                        float negJacobianToCanonical;
                        if (multiPointReSTIR.shouldSpatialReuse(i, negNeighboursValidMask))
                        {
                            negYiIntegrand = multiPointReSTIR.gPathTracer.shiftMappingGPath( negXi,
                                basePixel,
                                shiftPixel,
                                rtq,
                                negYi, negJacobianToCanonical );
                        }

                        float negMisWeight = 0.0f;
                        float negCC = 1.0f;
                        {
                            if (multiPointReSTIR.shouldSpatialReuse(i, negNeighboursValidMask) &&
                                shouldReuseBasedOnJacobian(negXi.innerJacobian, negYi.innerJacobian))
                            {
                                float negPHatToI = negJacobianToCanonical > 0.0f ? negTargetFunc(negXiIntegrand) / negJacobianToCanonical : 0.0f;
                                float mNegNum = negCi * negPHatToI;
                                float mNegDen = (negCTot - negCC) * negPHatToI +
                                                negCC * negTargetFunc(negYiIntegrand);
                                float negW = (negCTot - negCC) / negCTot;
                                negMisWeight = mNegDen > 0.0f ? negW * mNegNum / mNegDen : 0.0f;
                            }
                        }
                        if (multiPointReSTIR.shouldSpatialReuse(i, negNeighboursValidMask))
                        {
                            float negRisSelectU = sampleNext1D(risSg);
                            bool isSelected = negReservoir.addSpatialSample( finalPathBufferIndex,
                                                                             negYiIntegrand,
                                                                             negTargetFunc(negYiIntegrand),
                                                                             negJacobianToCanonical,
                                                                             negMisWeight,
                                                                             negXiReservoir.C,
                                                                             negXiReservoir.W,
                                                                             negRisSelectU );
                            negPath = isSelected ? negYi : negPath;
                        }
                    }
                    multiPointReSTIR.negPaths[finalPathBufferIndex] = negPath;
                }
            }

            negReservoir.finalizeSpatialReuse(multiPointReSTIR.gPathTracer.params.historyLength);
            multiPointReSTIR.negReservoirs[finalPathBufferIndex] = negReservoir;
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
