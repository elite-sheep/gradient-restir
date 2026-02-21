// Copyright @yucwang 2025

import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

import GPathTracer;
import Params;
import PathShift;
import RISDataTypes;
import StaticParams;
import Utils;

cbuffer CB
{
    PrimalSampleInitialSamples gPrimalIntegrator;
}

struct PrimalSampleInitialSamples
{
    GPathTracerParams params;
    StructuredBuffer<PathSampleRecord> basePaths;
    RWStructuredBuffer<PathSampleRecord> primalPaths;
    RWStructuredBuffer<PathReservoir> primalReservoirs;
    Texture1D<float4> basePathsRadiance;
    uint pathsPerPixel;

    void run(uint2 pixel)
    {
        uint pixelBufferIndex = getBufferOffset(pixel, params.frameDim);
        uint basePathBufferIndex = getBufferOffset(pixel, params.frameDim, kNumInitialSamples);
        uint primalPathBufferIndex = getBufferOffset(pixel, params.frameDim, pathsPerPixel);

        SampleGenerator risSg = SampleGenerator(pixel,
                                                params.seed * (kNumInitialSamples + 1));
        PathReservoir pathReservoir;
        pathReservoir.init();
        uint2 frameDim = params.frameDim;
        for (uint i = 0; i < kNumInitialSamples; ++i)
        {
            float4 color = float4(basePathsRadiance[basePathBufferIndex + i].xyz, 1.0f);
            float pathW = basePathsRadiance[basePathBufferIndex + i].z;

            float misWeight = 1.0f / kNumInitialSamples;
            float pHatI = targetFunc(color.xyz);
            pathReservoir.addCandidateSample(i,
                                             color.xyz,
                                             pHatI,
                                             misWeight,
                                             pathW,
                                             sampleNext1D(risSg));
        }
        primalPaths[primalPathBufferIndex] = basePaths[basePathBufferIndex + pathReservoir.pathID];
        pathReservoir.finalizeInitialCandidates();
        pathReservoir.pathID = primalPathBufferIndex;
        primalReservoirs[pixelBufferIndex] = pathReservoir;
    }
};


[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel < gPrimalIntegrator.params.frameDim))
    {
        gPrimalIntegrator.run(pixel);
    }
}
