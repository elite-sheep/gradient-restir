// Copyright @yucwang 2025

import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

import GPathTracer;
import Params;
import PathShift;
import PathState;
import PrimalReSTIR;
import RISDataTypes;
import StaticParams;
import Utils;

ParameterBlock<PrimalReSTIR> primalReSTIRSampler;
RWStructuredBuffer<PathReservoir> primalReservoirs;
RWStructuredBuffer<PathSampleRecord> primalPaths;
PrimalSpatialReuse gPrimalIntegrator;
uniform uint pathsPerPixel;

struct PrimalSpatialReuse
{
    void run(uint2 pixel)
    {
        uint bufferIndex = getBufferOffset(pixel, primalReSTIRSampler.gPathTracer.params.frameDim);

        SampleGenerator risSg = SampleGenerator(pixel,
                                        primalReSTIRSampler.gPathTracer.params.seed * (kNumInitialSamples + 4));
        uint startIndex = uint(sampleNext1D(risSg) * kNeighborOffsetCount);
        if (isInFrame(pixel, primalReSTIRSampler.gPathTracer.params.frameDim))
        {
            uint neighboursValidMask = 0;

            uint numSpatialNeighbours = 1;
    #if SHIFT_MAPPING_TYPE != 2
            VBuffer baseCentralPrimaryHit = primalReSTIRSampler.gPathTracer.vbuffer[bufferIndex];
    #endif
            for (int i = 0; i < numSpatialNeighbours; ++i)
            {
                int2 curXiPixel = primalReSTIRSampler.getNextNeighborPixel(startIndex, pixel, i);
                uint curXiOffset = getBufferOffset(curXiPixel, primalReSTIRSampler.gPathTracer.params.frameDim, pathsPerPixel);


                if (!isInFrame(curXiPixel, primalReSTIRSampler.gPathTracer.params.frameDim)
                    || primalReservoirs[curXiOffset].isNullSample)
                    continue;

                neighboursValidMask = setBit(neighboursValidMask, i);

    #if SHIFT_MAPPING_TYPE != 2
                uint curPrimalOffset = getBufferOffset(curXiPixel, primalReSTIRSampler.gPathTracer.params.frameDim);
                if (primalReSTIRSampler.shouldSpatialReuse(i, neighboursValidMask))
                {
                    VBuffer baseNeighbourPrimaryHitInfo = primalReSTIRSampler.gPathTracer.vbuffer[curPrimalOffset];
                    if (!primalReSTIRSampler.gPathTracer.isValidGeometry(baseCentralPrimaryHit, baseNeighbourPrimaryHitInfo))
                    {
                        neighboursValidMask = clearBit(neighboursValidMask, i);
                    }
                }
    #endif
            }

            float cTot = popcount32(neighboursValidMask) + 1.0f;
            // for (int j = 0; j < numSpatialNeighbours; ++j)
            // {
            //     int2 curXjPixel = primalReSTIRSampler.getNextNeighborPixel(startIndex, pixel, j);
            //     if (primalReSTIRSampler.shouldSpatialReuse(j, neighboursValidMask))
            //     {
            //         uint curBufferIndex = getBufferOffset(curXjPixel, primalReSTIRSampler.gPathTracer.params.frameDim);
            //         cTot += 1.0f;
            //     }
            // }

            uint primalPathOffset = getBufferOffset(pixel, primalReSTIRSampler.gPathTracer.params.frameDim, pathsPerPixel);
            PathSampleRecord path;
            RTQueryImpl rtq;
            PathReservoir reservoir;
            reservoir.init();
            {
                uint curPrimalOffset = getBufferOffset(pixel, primalReSTIRSampler.gPathTracer.params.frameDim, pathsPerPixel);
                PathReservoir curReservoir = primalReservoirs[curPrimalOffset];
                PathSampleRecord baseXi = primalPaths[curReservoir.pathID];
                float3 baseXiIntegrand = curReservoir.integrand;
                float3 baseYiIntegrand = baseXiIntegrand;
                float baseJacobianToCanonical = 1.0f;
                float baseMisWeight = 0.0f;
                float baseCC = 1.0f;
                float mPrimalNum = baseCC * targetFunc(baseYiIntegrand);
                baseMisWeight  = baseCC / cTot;

                for (int j = 0; j < numSpatialNeighbours; ++j)
                {
                    int2 curXjPixel = primalReSTIRSampler.getNextNeighborPixel(startIndex, pixel, j);

                    if (!primalReSTIRSampler.shouldSpatialReuse(j, neighboursValidMask)) continue;

                    uint shiftOffset = getBufferOffset(curXjPixel, primalReSTIRSampler.gPathTracer.params.frameDim);
                    {
                        PathSampleRecord baseXj;
                        float baseJacobianToJ;
                    #if SHIFT_MAPPING_TYPE == 2
                        float2 curXjFracPixelLoc = float2(curXjPixel);
                    #else
                        float2 curXjFracPixelLoc = float2(curXjPixel) + float2(0.5f);
                    #endif
                        // randomReplayShift(baseYi, baseXj, baseJacobianToJ);
                        float3 baseXjIntegrand = primalReSTIRSampler.gPathTracer.shiftMapping(baseXi, curXjFracPixelLoc,
                                                                                       primalReSTIRSampler.gPathTracer.vbuffer[shiftOffset],
                                                                                       rtq,
                                                                                       baseXj, baseJacobianToJ, false).xyz;

                        float mPrimalDen = mPrimalNum + (cTot - baseCC) * targetFunc(baseXjIntegrand) * baseJacobianToJ;
                        float w = 1.0f / cTot;
                        baseMisWeight += mPrimalDen > 0.0f ? w * mPrimalNum / mPrimalDen : 0.0f;
                    }
                }
                float risSelectU = sampleNext1D(risSg);
                bool isSelected = reservoir.addSpatialSample( primalPathOffset + pathsPerPixel - 1,
                                    baseYiIntegrand,
                                    targetFunc(baseYiIntegrand),
                                    baseJacobianToCanonical,
                                    baseMisWeight,
                                    curReservoir.C,
                                    curReservoir.W,
                                    risSelectU );
                path = isSelected ? baseXi : path;
            }

            for (int i = 0; i < numSpatialNeighbours; ++i)
            {
                bool shouldReuseXi = primalReSTIRSampler.shouldSpatialReuse(i, neighboursValidMask);
                int2 curXiPixel = primalReSTIRSampler.getNextNeighborPixel(startIndex, pixel, i);

                if (!primalReSTIRSampler.shouldSpatialReuse(i, neighboursValidMask)) continue;

                uint curPrimalOffset = getBufferOffset(curXiPixel, primalReSTIRSampler.gPathTracer.params.frameDim, pathsPerPixel);

                PathReservoir curSpatialReservoir = primalReservoirs[curPrimalOffset];
                PathSampleRecord baseXi = primalPaths[curSpatialReservoir.pathID];
                float3 baseXiIntegrand = curSpatialReservoir.integrand;
    #if SHIFT_MAPPING_TYPE == 2
                float2 curYiFracPixelLoc = float2(pixel);
    #else
                float2 curYiFracPixelLoc = float2(pixel) + float2(0.5f);
    #endif
                PathSampleRecord baseYi;
                float3 baseYiIntegrand;
                float baseJacobianToCanonical;
                {
                    baseYiIntegrand = primalReSTIRSampler.gPathTracer.shiftMapping( baseXi, curYiFracPixelLoc,
                        primalReSTIRSampler.gPathTracer.vbuffer[bufferIndex],
                        rtq,
                        baseYi, baseJacobianToCanonical, false );
                }

                float baseMisWeight = 0.0f;
                float baseCC = 1.0f;
                {
                    float basePHatToI = baseJacobianToCanonical > 0.0f ? targetFunc(baseXiIntegrand) / baseJacobianToCanonical : 0.0f;
                    float mPrimalNum = (cTot - baseCC) * basePHatToI;
                    float mPrimalDen = mPrimalNum + baseCC * targetFunc(baseYiIntegrand);
                    float w = 1.0f / cTot;
                    baseMisWeight = mPrimalDen > 0.0f ? w * mPrimalNum / mPrimalDen : 0.0f;
                }

                {
                    float risSelectU = sampleNext1D(risSg);
                    bool isSelected = reservoir.addSpatialSample( primalPathOffset + pathsPerPixel - 1,
                                        baseYiIntegrand,
                                        targetFunc(baseYiIntegrand),
                                        baseJacobianToCanonical,
                                        baseMisWeight,
                                        curSpatialReservoir.C,
                                        curSpatialReservoir.W,
                                        risSelectU );
                    path = isSelected ? baseYi : path;
                }
            }

            reservoir.finalizeSpatialReuse(primalReSTIRSampler.gPathTracer.params.historyLength);
            primalPaths[primalPathOffset + pathsPerPixel - 1] = path;
            primalReSTIRSampler.primalReservoirs[primalPathOffset + pathsPerPixel - 1] = reservoir;
        }
    }
};


[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel < primalReSTIRSampler.gPathTracer.params.frameDim))
    {
        gPrimalIntegrator.run(pixel);
    }
}
