// Copyright @yucwang 2025

import Scene.RaytracingInline;
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

// ParameterBlock<PrimalReSTIR> primalReSTIRSampler;
uniform GPathTracerParams params;
uniform uint pathsPerPixel;
PrimalTemporalReuse gPrimalIntegrator;
ParameterBlock<GPathTracer> gPathTracer;
StructuredBuffer<PathReservoir> temporalReservoirs;
RWStructuredBuffer<PathSampleRecord> basePaths;
RWStructuredBuffer<PathSampleRecord> primalPaths;
RWStructuredBuffer<PathReservoir> curReservoirs;
RWTexture2D<float2> temporalFracPixelLocs;
RWTexture2D<uint>  canFindTemporalNeighbours;
StructuredBuffer<float4> basePathsRadiance;

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
}

struct PrimalTemporalReuse
{
    void run(uint2 pixel)
    {
        PathSampleRecord path;
        uint primalPathBufferIndex = getBufferOffset(pixel, params.frameDim, pathsPerPixel);
        {
            uint pixelBufferIndex = getBufferOffset(pixel, params.frameDim);
            uint basePathBufferIndex = getBufferOffset(pixel, params.frameDim, kNumInitialSamples);

            SampleGenerator risSg = SampleGenerator(pixel,
                                                    params.seed * (kNumInitialSamples + 1));
            PathReservoir pathReservoir;
            pathReservoir.init();
            uint2 frameDim = params.frameDim;
            for (uint i = 0; i < kNumInitialSamples; ++i)
            {
                float4 color = float4(basePathsRadiance[basePathBufferIndex + i].xyz, 1.0f);
                float pathW = basePathsRadiance[basePathBufferIndex + i].w;

                float misWeight = 1.0f / kNumInitialSamples;
                float pHatI = targetFunc(color.xyz);
                pathReservoir.addCandidateSample(i,
                                                 color.xyz,
                                                 pHatI,
                                                 misWeight,
                                                 pathW,
                                                 sampleNext1D(risSg));
            }
            path = basePaths[basePathBufferIndex + pathReservoir.pathID];
            pathReservoir.finalizeInitialCandidates();
            pathReservoir.pathID = primalPathBufferIndex;
            curReservoirs[primalPathBufferIndex] = pathReservoir;
        }

        if (kUseTemporalReuse)
        {
            SampleGenerator risSg = SampleGenerator(pixel, params.seed * (kNumInitialSamples + 2));
            uint2 frameDim = params.frameDim;
            uint bufferIndex = getBufferOffset(pixel, frameDim);
            uint primalPathBufferIndex = getBufferOffset(pixel, frameDim, pathsPerPixel);

            VBuffer curPrimaryHit;
            VBuffer prevPrimaryHit;
            PathReservoir temporalReservoir = temporalReservoirs[bufferIndex];
            float2 temporalFracPixelLoc = temporalFracPixelLocs[pixel];
            bool canFindTemporalNeighbour = !temporalReservoir.isNullSample;
            // if (!canFindTemporalNeighbour)
            // {
            //     return;
            // }
        #if SHIFT_MAPPING_TYPE == 2
            float2 screenMotion = gPathTracer.motionVectors[pixel] * float2(frameDim);
            float2 prevFracCenter = float2(pixel) + float2(0.5f, 0.5f) + screenMotion;
            float2 prevFracMin = prevFracCenter - float2(0.5f);
        #else
            float2 motionVector = gPathTracer.motionVectors[pixel];
            int2 prevPixel = pixel;
            prevPixel = int2(pixel + motionVector * frameDim + sampleNext2D(risSg));
            uint prevBufferIndex = getBufferOffset(prevPixel, frameDim);
            curPrimaryHit = gPathTracer.vbuffer[bufferIndex];
            prevPrimaryHit = gPathTracer.prevVBuffer[prevBufferIndex];
            if (!gPathTracer.isValidGeometry(curPrimaryHit, prevPrimaryHit))
            {
                return;
            }
        #endif

            RTQueryImpl rtq;
            if (isInFrame(pixel, frameDim) && canFindTemporalNeighbour)
            {
                float cTot = temporalReservoir.C + 1.0f;
                PathReservoir reservoir;
                // Reset the RIS status
                reservoir.init();
                {
                    PathReservoir curReservoir = curReservoirs[primalPathBufferIndex];
                    PathSampleRecord xi = path;
                    PathSampleRecord xj;
                    float jacobianToJ;
        #if SHIFT_MAPPING_TYPE == 2
                    float2 xjFracPixelLoc = prevFracMin;
        #else
                    float2 xjFracPixelLoc = float2(prevPixel) + float2(0.5f);
        #endif
                    float3 xiF = curReservoir.integrand;
                    float misWeight = 1.0f;
                    float3 xjF = gPathTracer.shiftMapping( xi,
                                                           xjFracPixelLoc,
                                                           prevPrimaryHit,
                                                           rtq, xj, jacobianToJ, true );

                    float denom = targetFunc(xiF) +
                                  temporalReservoir.C * targetFunc(xjF) * jacobianToJ;
                    float num =   targetFunc(xiF);

                    bool isProjInScreen = isInFrameFloat(xjFracPixelLoc + xi.pathRcInfo.fracPixelLoc, frameDim);
                    misWeight = (isProjInScreen && denom > 0.0f) ? num / denom : 1.0f;
                    reservoir.addTemporalSample(primalPathBufferIndex,
                                                xiF,
                                                targetFunc(xiF),
                                                1.0f,
                                                misWeight,
                                                1.0f,
                                                curReservoir.W,
                                                sampleNext1D(risSg) );
                }

                {
                    PathSampleRecord xi = primalPaths[temporalReservoir.pathID];
                    PathSampleRecord yi;
                    float jacobianToCanonical;
        #if SHIFT_MAPPING_TYPE == 2
                    float2 yiFracPixelLoc = float2(pixel);
                    xi.pathRcInfo.fracPixelLoc = temporalFracPixelLoc;
        #else
                    float2 yiFracPixelLoc = pixel + float2(0.5f);
        #endif

                    float3 xiF = temporalReservoir.integrand;
                    float3 yiF = gPathTracer.shiftMapping( xi,
                                                           yiFracPixelLoc,
                                                           curPrimaryHit,
                                                           rtq,
                                                           yi, jacobianToCanonical, false ) .xyz;

                    float num =   jacobianToCanonical > 0.0f ? temporalReservoir.C * targetFunc(xiF) / jacobianToCanonical : 0.0f;
                    float denom = targetFunc(yiF) + num;
                    float misWeight = denom > 0.0f ? num / denom : 0.0f;
                    float risSelectU = sampleNext1D(risSg);
                    bool isTemporalSelected = reservoir.addTemporalSample( primalPathBufferIndex,
                                                                           yiF,
                                                                           targetFunc(yiF),
                                                                           jacobianToCanonical,
                                                                           misWeight,
                                                                           temporalReservoir.C,
                                                                           temporalReservoir.W,
                                                                           risSelectU );
                    path = isTemporalSelected ? yi : path;
                }
                reservoir.finalizeTemporalReuse(gPathTracer.params.historyLength);
                curReservoirs[primalPathBufferIndex] = reservoir;
            }
        }
        primalPaths[primalPathBufferIndex] = path;
    }
};


[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    printSetPixel(dispatchThreadId.xy);
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel < gPathTracer.params.frameDim))
    {
        gPrimalIntegrator.run(pixel);
    }
}
