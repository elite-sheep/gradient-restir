// Copyright @yucwang 2024

import Scene.RaytracingInline;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

import CameraUtils;
import GPathTracer;
import PathShift;
import PathState;
import StaticParams;
import TracingUtils;
import Utils;

ParameterBlock<GPathTracer> gGPathTracer;
RWStructuredBuffer<VBuffer> primaryHitInfo;
RWStructuredBuffer<PathSampleRecord> basePaths;
RWTexture2D<float4> primaryEmission;
RWStructuredBuffer<float4> basePathsRadiance;
RWStructuredBuffer<float4> dxShiftedPathsRadiance;
RWStructuredBuffer<float4> dyShiftedPathsRadiance;
Scheduler gScheduler;

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

struct Scheduler
{
    void run(uint2 pixel)
    {
        printSetPixel(pixel);

        uint2 frameDim = gGPathTracer.params.frameDim;
        uint bufferIndex = getBufferOffset(pixel, frameDim, kNumInitialSamples);

        RTQueryImpl rtq;
        float3 emission;
        VBuffer mPrimaryHitInfo;
        gGPathTracer.generatePrimayHit(pixel, rtq, mPrimaryHitInfo);
        primaryHitInfo[bufferIndex] = mPrimaryHitInfo;
        for (int i = 0; i < kNumInitialSamples; ++i)
        {
            PathSampleRecord path;
            path.initRandomSeed.xy = pixel;
            path.initRandomSeed.z = gGPathTracer.params.seed * (i + 1);
            float4 L = gGPathTracer.generatePath(pixel, mPrimaryHitInfo, 0, false,
                                        rtq, path);
            basePaths[bufferIndex + i].update(path);
            basePathsRadiance[bufferIndex + i] = L;

            {
                float dxJacobian1;
                float3 dxIntegrand = gGPathTracer.areaReSTIRShiftMappingLight(path,
                                                                              float2(pixel + uint2(1, 0)),
                                                                              false,
                                                                              rtq,
                                                                              dxJacobian1).xyz;
                dxShiftedPathsRadiance[2 * (bufferIndex + i)] = float4(dxIntegrand, dxJacobian1);
            }
            {
                float dxJacobian1;
                float3 dxIntegrand = gGPathTracer.areaReSTIRShiftMappingLight(path,
                                                                              float2(pixel - uint2(1, 0)),
                                                                              false,
                                                                              rtq,
                                                                              dxJacobian1).xyz;
                dxShiftedPathsRadiance[2 * (bufferIndex + i) + 1] = float4(dxIntegrand, dxJacobian1);
            }
            {
                float dyJacobian1;
                float3 dyIntegrand = gGPathTracer.areaReSTIRShiftMappingLight(path,
                                                                              float2(pixel + uint2(0, 1)),
                                                                              false,
                                                                              rtq,
                                                                              dyJacobian1).xyz;
                dyShiftedPathsRadiance[2 * (bufferIndex + i)] = float4(dyIntegrand, dyJacobian1);
            }
            {
                float dyJacobian1;
                float3 dyIntegrand = gGPathTracer.areaReSTIRShiftMappingLight(path,
                                                                              float2(pixel - uint2(0, 1)),
                                                                              false,
                                                                              rtq,
                                                                              dyJacobian1).xyz;
                dyShiftedPathsRadiance[2 * (bufferIndex + i) + 1] = float4(dyIntegrand, dyJacobian1);
            }
        }
    }
};

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel < gGPathTracer.params.frameDim))
    {
        gScheduler.run(pixel);
    }
}
