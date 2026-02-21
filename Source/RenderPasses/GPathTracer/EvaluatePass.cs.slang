// Copyright @yucwang 2024

import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

import DiffIntegrator;
import GPathTracer;
import PathShift;
import PrimalReSTIR;
import RISDataTypes;
import StaticParams;

ParameterBlock<GPathTracer> gGPathTracer;
ParameterBlock<PrimalReSTIR> gPrimalIntegrator;
ParameterBlock<DiffIntegrator> gDXIntegrator;
ParameterBlock<DiffIntegrator> gDYIntegrator;
Scheduler gScheduler;

// Outputs
RWTexture2D<float4> outputPrimal;
RWTexture2D<float4> outputDX;
RWTexture2D<float4> outputDY;

struct Scheduler
{
    void run(uint2 pixel)
    {
        uint bufferIndex = gGPathTracer.params.frameDim.x * pixel.y + pixel.x;
        printSetPixel(pixel);
        outputPrimal[pixel] = gPrimalIntegrator.evaluate(pixel);
        // DX Pass
        outputDX[pixel] = gDXIntegrator.evaluate(pixel, uint2(1, 0));
        // DY Pass
        outputDY[pixel] = gDYIntegrator.evaluate(pixel, uint2(0, 1));
    }
};

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel < gGPathTracer.params.frameDim))
    {
        gScheduler.run(pixel);
    }
}
