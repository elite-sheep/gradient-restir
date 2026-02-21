// Copyright @yucwang 2025

import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

import GPathTracer;
import Params;
import PathShift;
import PathState;
import RISDataTypes;
import StaticParams;
import Utils;

ParameterBlock<GPathTracer> gPathTracer;
StructuredBuffer<PathReservoir> primalReservoirs;
RWStructuredBuffer<PathSampleRecord> primalPaths;
RWStructuredBuffer<PathReservoir> sampledTemporalReservoirs;
uniform GPathTracerParams params;
uniform uint pathsPerPixel;
PrimalTemporalReuseLookForNeighbours gPrimalTemporalLookForNeighbours;
RWTexture2D<float2> temporalFracPixelLocs;
RWTexture2D<uint> canFindTemporalNeighbours;

struct PrimalTemporalReuseLookForNeighbours
{
    bool lookForTemporalNeighbour(uint2 pixel,
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

        canFindTemporalNeighbours[pixel] &= 0xFFFFFFFE;
        PathReservoir pathReservoir;
        pathReservoir.init();
        float2 fracPixelLoc = float2(0.0f);
        for (int i = 0; i < 4; ++i)
        {
            int2 offset = int2(i & 1, (i >> 1) & 1);
            int2 curPixel = originPixel + offset;
            uint primalPathBufferIndex = getBufferOffset(curPixel, frameDim, pathsPerPixel);
            PathReservoir curReservoir = primalReservoirs[primalPathBufferIndex + pathsPerPixel - 1];
            uint curPathID = curReservoir.pathID;
            if (isInFrame(curPixel, frameDim))
            {
                float2 curFracPixelLoc = curPixel +
                                         primalPaths[curPathID].pathRcInfo.fracPixelLoc;
                float misWeight = (all(curFracPixelLoc > pixelLow) && all(curFracPixelLoc < pixelHigh)) ? 1.0f : 0.0f;
                float3 curXiF = curReservoir.integrand;
                bool isSelected = pathReservoir.addTemporalSample( curPathID,
                    curXiF, targetFunc(curXiF), 1.0f, misWeight,
                    curReservoir.C * bilinearWeights[i],
                    curReservoir.W, sampleNext1D(sg) );
                fracPixelLoc = isSelected ? curFracPixelLoc - pixelLow : fracPixelLoc;
                foundAValidNeigbour = isSelected ? true : foundAValidNeigbour;
            }
        }
        pathReservoir.finalizeTemporalReuse(params.historyLength);
        uint pixelBufferIndex = getBufferOffset(pixel, frameDim);
        sampledTemporalReservoirs[pixelBufferIndex] = pathReservoir;
        temporalFracPixelLocs[pixel] = fracPixelLoc;
        canFindTemporalNeighbours[pixel] = foundAValidNeigbour ? (canFindTemporalNeighbours[pixel] | 0x1) :
                                           canFindTemporalNeighbours[pixel];
        print("primalFracPixelLoc:", fracPixelLoc);
        print("primalCanFindTemporalNeighbour:", pathReservoir.isNullSample);

        return foundAValidNeigbour;
    }

    void run(uint2 pixel)
    {
        SampleGenerator risSg = SampleGenerator(pixel, params.seed * (kNumInitialSamples + 2));
        uint2 frameDim        = params.frameDim;
        uint bufferIndex = getBufferOffset(pixel, frameDim);

        bool canFindTemporalNeighbour = true;
#if SHIFT_MAPPING_TYPE == 2
        float2 screenMotion = gPathTracer.motionVectors[pixel] * float2(frameDim);
        float2 prevFracCenter = float2(pixel) + float2(0.5f, 0.5f) + screenMotion;
        float2 prevFracMin = prevFracCenter - float2(0.5f);
        if (!isInFrame(prevFracMin, frameDim))
        {
            print("prevFracMin:", prevFracMin);
            print("frameDim:", frameDim);
            return;
        }
        int2 prevPixel;
        lookForTemporalNeighbour(pixel, prevFracCenter, frameDim, risSg);
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
        sampledTemporalReservoirs[bufferIndex] = primalReservoirs[prevBufferIndex + pathsPerPixel - 1];
#endif
    }
};

[numthreads(8, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    printSetPixel(dispatchThreadId.xy);
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel < params.frameDim))
    {
        gPrimalTemporalLookForNeighbours.run(pixel);
    }
}
