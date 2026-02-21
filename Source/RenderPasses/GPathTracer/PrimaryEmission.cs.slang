// Copyright @yucwang 2024

import Scene.RaytracingInline;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

import CameraUtils;
import GPathTracer;
import PathShift;
import StaticParams;
import TracingUtils;
import Utils;

ParameterBlock<GPathTracer> gGPathTracer;
RWTexture2D<float4> primaryEmission;
RWTexture2D<float4> albedo;
RWTexture2D<float4> normalAndDepth;
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
        uint bufferIndex = getBufferOffset(pixel, gGPathTracer.params.frameDim);
        float2 fracPixelLoc = pixel + float2(0.5f, 0.5f);

        RTQueryImpl rtq;
        Ray cameraRay = computeRayPinhole(gScene.camera.data, fracPixelLoc, gGPathTracer.params.frameDim);
        HitInfo primaryHitInfo;
        float primaryHitT;
        float hitT;
        bool isHit = rtq.traceRay(cameraRay, primaryHitInfo, hitT);
        let lod = ExplicitLodTextureSampler(0.0f);
        ShadingData primaryShadingData = loadShadingData(primaryHitInfo, cameraRay, lod);
        let mi = gScene.materials.getMaterialInstance(primaryShadingData, lod);
        primaryEmission[pixel] = float4(mi.getProperties(primaryShadingData).emission, 1.0f);
        albedo[pixel] = float4(mi.getProperties(primaryShadingData).diffuseReflectionAlbedo, 1.0f);
        normalAndDepth[pixel].xyz = primaryShadingData.frame.N;

        float4 curPosH = mul(gScene.camera.data.viewProjMatNoJitter,
                            float4(primaryShadingData.posW, 1.f));
        normalAndDepth[pixel].w = curPosH.z / curPosH.w;
    }
};

[numthreads(32, 32, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel < gGPathTracer.params.frameDim))
    {
        gScheduler.run(pixel);
    }
}
