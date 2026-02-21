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
RWTexture2D<uint2> neighbourPixels;

struct MultiPointSpatialReuse
{
    void run(uint2 basePixel)
    {
        uint2 frameDim = gPathTracer.params.frameDim;
        uint2 shift = (shiftDir == 0) ? uint2(1, 0) : uint2(0, 1);
        uint2 shiftPixel = basePixel + shift;
        SampleGenerator sg = SampleGenerator(basePixel, multiPointReSTIR.gPathTracer.params.seed * (kNumInitialSamples + 5));
        uint startIndex = uint(sampleNext1D(sg) * kNeighborOffsetCount);
        bool shouldReuse = multiPointReSTIR.shouldSpatialReuse(basePixel, basePixel + shift) && kNumSpatialNeighbours > 0;
        int2 neighbourPixel = shouldReuse ? multiPointReSTIR.getNeibourPixel(basePixel, shift, startIndex) : basePixel;
        neighbourPixels[basePixel] = neighbourPixel;
    }
};

[numthreads(32, 32, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel < gPathTracer.params.frameDim))
    {
        gGradientIntegrator.run(pixel);
    }
}
