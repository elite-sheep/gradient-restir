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

MultiPointTemporalLookForNeighbour gGradientIntegrator;
ParameterBlock<GPathTracer> gPathTracer;
ParameterBlock<MultiPointReSTIR> multiPointReSTIR;
uniform uint shiftDir;
uniform uint pathsPerPixel;

struct MultiPointTemporalLookForNeighbour
{
    bool negLookForTemporalNeighbour(uint2 pixel,
                                     float2 originCenterFracPixel,
                                     uint2 frameDim,
                                     inout SampleGenerator sg)
    {
        int2 originPixel = int2(floor(originCenterFracPixel - float2(0.5f, 0.5f)));
        float2 pixelLow = originCenterFracPixel - float2(0.5f, 0.5f);
        float2 pixelHigh = originCenterFracPixel + float2(0.5f, 0.5f);
        bool foundAValidNeigbour = false;

        float4 bilinearWeights;
        computeBilinearWeights(pixelLow - originPixel, bilinearWeights);

        PathReservoir pathReservoir;
        pathReservoir.init();
        float2 fracPixelLoc = float2(0.0f);
        for (int i = 0; i < 4; ++i)
        {
            int2 offset = int2(i & 1, (i >> 1) & 1);
            int2 curPixel = originPixel + offset;
            uint primalPathBufferIndex = getBufferOffset(curPixel, frameDim, pathsPerPixel);
            PathReservoir curReservoir = multiPointReSTIR.negReservoirs[primalPathBufferIndex + pathsPerPixel - 1];
            uint curPathID = curReservoir.pathID;
            if (isInFrame(curPixel, frameDim) && !curReservoir.isNullSample)
            {
                float2 curFracPixelLoc = curPixel +
                                         multiPointReSTIR.negPaths[curPathID].pathRcInfo.fracPixelLoc;
                float misWeight = (all(curFracPixelLoc > pixelLow) && all(curFracPixelLoc < pixelHigh)) ? 1.0f : 0.0f;
                float3 curXiF = curReservoir.integrand;
                bool isSelected = pathReservoir.addTemporalSample( curPathID,
                    curXiF, negTargetFunc(curXiF), 1.0f, misWeight,
                    curReservoir.C,
                    curReservoir.W, sampleNext1D(sg) );
                fracPixelLoc = isSelected ? curFracPixelLoc - pixelLow : fracPixelLoc;
                foundAValidNeigbour = isSelected ? true : foundAValidNeigbour;
            }
        }
        pathReservoir.finalizeTemporalReuse(gPathTracer.params.historyLength);
        uint pixelBufferIndex = getBufferOffset(pixel, frameDim);
        multiPointReSTIR.negSampledTemporalReservoirs[pixelBufferIndex] = pathReservoir;
        multiPointReSTIR.negTemporalFracPixelLocs[pixel] = fracPixelLoc;

        return foundAValidNeigbour;
    }

    bool posLookForTemporalNeighbour(uint2 pixel,
                                     float2 originCenterFracPixel,
                                     uint2 frameDim,
                                     inout SampleGenerator sg)
    {
        int2 originPixel = int2(floor(originCenterFracPixel - float2(0.5f, 0.5f)));
        float2 pixelLow = originCenterFracPixel - float2(0.5f, 0.5f);
        float2 pixelHigh = originCenterFracPixel + float2(0.5f, 0.5f);
        bool foundAValidNeigbour = false;

        float4 bilinearWeights;
        computeBilinearWeights(pixelLow - originPixel, bilinearWeights);

        PathReservoir pathReservoir;
        pathReservoir.init();
        float2 fracPixelLoc = float2(0.0f);
        for (int i = 0; i < 4; ++i)
        {
            int2 offset = int2(i & 1, (i >> 1) & 1);
            int2 curPixel = originPixel + offset;
            uint primalPathBufferIndex = getBufferOffset(curPixel, frameDim, pathsPerPixel);
            PathReservoir curReservoir = multiPointReSTIR.posReservoirs[primalPathBufferIndex + pathsPerPixel - 1];
            uint curPathID = curReservoir.pathID;
            if (isInFrame(curPixel, frameDim) && !curReservoir.isNullSample)
            {
                float2 curFracPixelLoc = curPixel +
                                         multiPointReSTIR.posPaths[curPathID].pathRcInfo.fracPixelLoc;
                float misWeight = (all(curFracPixelLoc > pixelLow) && all(curFracPixelLoc < pixelHigh)) ? 1.0f : 0.0f;
                float3 curXiF = curReservoir.integrand;
                bool isSelected = pathReservoir.addTemporalSample( curPathID,
                    curXiF, posTargetFunc(curXiF), 1.0f, misWeight,
                    curReservoir.C,
                    curReservoir.W, sampleNext1D(sg) );
                fracPixelLoc = isSelected ? curFracPixelLoc - pixelLow : fracPixelLoc;
                foundAValidNeigbour = isSelected ? true : foundAValidNeigbour;
            }
        }
        pathReservoir.finalizeTemporalReuse(gPathTracer.params.historyLength);
        uint pixelBufferIndex = getBufferOffset(pixel, frameDim);
        multiPointReSTIR.posSampledTemporalReservoirs[pixelBufferIndex] = pathReservoir;
        multiPointReSTIR.posTemporalFracPixelLocs[pixel] = fracPixelLoc;
        print("posFracPixelLoc:", fracPixelLoc);

        return foundAValidNeigbour;
    }

    void run(uint2 pixel)
    {
        SampleGenerator risSg = SampleGenerator(pixel, gPathTracer.params.seed * (kNumInitialSamples + 2));
        uint2 frameDim        = gPathTracer.params.frameDim;
        uint bufferIndex = getBufferOffset(pixel, frameDim);
        uint2 shift = (shiftDir == 0) ? uint2(1, 0) : uint2(0, 1);
        uint2 shiftPixel = pixel + shift;

        bool canFindTemporalNeighbour = true;
#if SHIFT_MAPPING_TYPE == 2
        float2 motionVector;
        if (multiPointReSTIR.primaryDepth[pixel] < multiPointReSTIR.primaryDepth[shiftPixel])
        {
            motionVector = gPathTracer.motionVectors[pixel];
        }
        else
        {
            motionVector = gPathTracer.motionVectors[shiftPixel];
        }
        float2 screenMotion = motionVector * float2(frameDim);
        float2 prevFracCenter = float2(pixel) + float2(0.5f, 0.5f) + screenMotion;
        float2 prevFracMin = prevFracCenter - float2(0.5f);
        if (!isInFrame(prevFracMin, frameDim))
        {
            print("prevFracMin:", prevFracMin);
            print("frameDim:", frameDim);
            return;
        }
        posLookForTemporalNeighbour(pixel, prevFracCenter, frameDim, risSg);
        negLookForTemporalNeighbour(pixel, prevFracCenter, frameDim, risSg);
#else
        float2 motionVector = gPathTracer.motionVectors[pixel];
        int2 prevPixel = pixel;
        prevPixel = int2(pixel + motionVector * frameDim + sampleNext2D(risSg));
        uint prevBufferIndex = getBufferOffset(prevPixel, frameDim, pathsPerPixel);
        VBuffer curPrimaryHit;
        VBuffer prevPrimaryHit;
        curPrimaryHit = gPathTracer.vbuffer[bufferIndex];
        prevPrimaryHit = gPathTracer.prevVBuffer[prevBufferIndex];
        if (!gPathTracer.isValidGeometry(curPrimaryHit, prevPrimaryHit))
        {
            return;
        }
        multiPointReSTIR.posSampledTemporalReservoirs[prevBufferIndex + pathsPerPixel - 1] =
            multiPointReSTIR.posReservoirs[prevBufferIndex + pathsPerPixel - 1];
        multiPointReSTIR.negSampledTemporalReservoirs[prevBufferIndex + pathsPerPixel - 1] =
            multiPointReSTIR.negReservoirs[prevBufferIndex + pathsPerPixel - 1];
#endif
    }

};

[numthreads(8, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel < gPathTracer.params.frameDim))
    {
        gGradientIntegrator.run(pixel);
    }
}
