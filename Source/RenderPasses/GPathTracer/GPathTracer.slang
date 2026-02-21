// Copyright @yucwang 2024

#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.EmissiveLightSampler;
import Scene.Camera.CameraData;
import Scene.Raytracing;
import Scene.Intersection;
import Scene.ShadingData;
import Scene.SceneTypes;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Geometry.GeometryHelpers;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;

import CameraUtils;
import Params;
import PathShift;
import PathState;
import RISDataTypes;
import StaticParams;
import TracingUtils;
import Utils;

struct GPathTracer
{
    static const float kPixelGaussianSigma = 0.5f;

    // Config parameters
    GPathTracerParams params;

    // Emissive samplers
    EmissiveLightSampler emissiveLightSampler;
    CameraData prevCameraData;

    // Inputs
    StructuredBuffer<VBuffer> vbuffer;
    StructuredBuffer<VBuffer> prevVBuffer;

    Texture2D<PackedHitInfo> vbufferCenter;
    Texture2D<float4> viewDir;
    Texture2D<float2> motionVectors;
    Texture2D<float2> subPixelUV;

    // Outputs
    RWTexture2D<float4> outputPrimal;
    RWTexture2D<float4> outputDX;
    RWTexture2D<float4> outputDY;

    interface IRTQuery
    {
        bool traceRay(const Ray ray, out HitInfo hit, out float hitT);
        bool traceVisibilityRay(const Ray ray);
    };

    ShadingData getPixelShadingData( HitInfo hitInfo, int2 pixel)
    {
        ShadingData sd = {};
        Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        if (hitInfo.isValid())
        {
            sd = loadShadingData(hitInfo, ray, ExplicitLodTextureSampler(0));
        }
        return sd;
    }

    bool isValidGeometry(VBuffer vBuffer1, VBuffer vBuffer2)
    {
        ExplicitLodTextureSampler lod = ExplicitLodTextureSampler(0.0f);
        HitInfo centralHit = vBuffer1.primaryHitInfo;
        HitInfo neighborHit = vBuffer2.primaryHitInfo;
        // TODO: update isPrevFrame
        ShadingData centralShadingData = loadShadingData(centralHit, vBuffer1.cameraRay, lod);
        ShadingData neighbourShadingData = loadShadingData(neighborHit, vBuffer2.cameraRay, lod);
        float centralDist = distance(vBuffer1.cameraRay.origin, centralShadingData.posW);
        float neighborDist = distance(vBuffer2.cameraRay.origin, neighbourShadingData.posW);

        return dot(centralShadingData.frame.N, neighbourShadingData.frame.N) >= 0.5f &&
               abs(centralDist - neighborDist) < 0.1f * centralDist;
    }

    float3 shiftAndComputeGPath<RTQuery : IRTQuery>(inout PathSampleRecord path,
                                                     const float2 basePixel,
                                                     const float3 baseIntegrand,
                                                     const float2 shiftPixel,
                                                     RTQuery rtq,
                                                     bool isPrevFrame = false)
    {
        uint k = path.index;
        float s = (k == 0) ? 1.0f : -1.0f;
        float2 sampleBasePixel = (k == 0) ? basePixel : shiftPixel;
        float2 sampleShiftPixel = (k == 0) ? shiftPixel : basePixel;

        float3 shift = areaReSTIRShiftMappingLight(path,
                                                   sampleShiftPixel,
                                                   isPrevFrame,
                                                   rtq,
                                                   path.innerJacobian).xyz;

        float denom = targetFunc(baseIntegrand) + targetFunc(shift) * path.innerJacobian;
        float w = denom > 0.0f ? targetFunc(baseIntegrand) / denom : 0.0f;

        return s * w * (shift * path.innerJacobian - baseIntegrand);
    }

    float3 shiftMappingGPath<RTQuery : IRTQuery>(const PathSampleRecord originPath,
                                                 const float2 basePixel,
                                                 const float2 shiftPixel,
                                                 RTQuery rtq,
                                                 out PathSampleRecord shiftedPath,
                                                 out float jacobian,
                                                 bool isPrevFrame = false )
    {
        uint k = originPath.index;
        float s = (k == 0) ? 1.0f : -1.0f;
        float2 sampleBasePixel = (k == 0) ? basePixel : shiftPixel;
        float2 sampleShiftPixel = (k == 0) ? shiftPixel : basePixel;
        // VBuffer basePrimaryHitInfo = (k == 0) ? primaryHitInfo1 : primaryHitInfo2;
        // VBuffer shiftPrimaryHitInfo = (k == 0) ? primaryHitInfo2 : primaryHitInfo1;

        float3 base = areaReSTIRShiftMapping(originPath,
                                   sampleBasePixel,
                                   isPrevFrame,
                                   rtq,
                                   shiftedPath,
                                   jacobian).xyz;

        float innerJacobian;
        float3 shift = areaReSTIRShiftMappingLight(shiftedPath,
                                                   sampleShiftPixel,
                                                   isPrevFrame,
                                                   rtq,
                                                   innerJacobian).xyz;

        float denom = targetFunc(base) + targetFunc(shift) * innerJacobian;
        float w = denom > 0.0f ? targetFunc(base) / denom : 0.0f;

        shiftedPath.index = originPath.index;
        shiftedPath.innerJacobian = innerJacobian;

        return s * w * (shift * innerJacobian - base);
    }

    float3 shiftMapping<RTQuery : IRTQuery>(const PathSampleRecord originPath,
                                            const float2 pixel,
                                            const VBuffer primaryHitInfo,
                                            RTQuery rtq,
                                            out PathSampleRecord shiftedPath,
                                            out float jacobian,
                                            bool isPrevFrame = false)
    {
        float3 result;
#if SHIFT_MAPPING_TYPE == 0
            result = hybridShiftMapping(originPath, uint2(pixel),
                                        primaryHitInfo, rtq,
                                        shiftedPath, jacobian).xyz;
#elif SHIFT_MAPPING_TYPE == 1
            randomReplayShift(originPath, shiftedPath, jacobian);
            result = randomPlayPath(pixel, primaryHitInfo, rtq, shiftedPath).xyz;
#elif SHIFT_MAPPING_TYPE == 2
            result = areaReSTIRShiftMapping(originPath, pixel,
                                            isPrevFrame, rtq, shiftedPath, jacobian).xyz;
#endif
            if (min(jacobian, 1.0 / jacobian) < 0.1f)
            {
                jacobian = 0.0f;
                result = float3(0.0f);
            }
        return result;
    }

    float4 generateGPath<RTQuery : IRTQuery>(const uint2 basePixel,
                                             const VBuffer basePrimaryHitInfo,
                                             const uint2 shiftPixel,
                                             const VBuffer shiftPrimaryHitInfo,
                                             RTQuery rtq,
                                             inout PathSampleRecord pathSampleRecord)
    {
        float4 base = generatePath(basePixel, basePrimaryHitInfo, 0u, false, rtq, pathSampleRecord);

        PathSampleRecord shiftPathSample;
        float jacobian;

        float3 shift = shiftMapping(pathSampleRecord,
                                    shiftPixel, shiftPrimaryHitInfo,
                                    rtq,
                                    shiftPathSample, jacobian);
        // float w = luminance(base.xyz) / (luminance(base.xyz) + luminance(shift.xyz));
        // float w = (base.x + shift.x) > 0.0f ? base.x / (base.x + shift.x) : 0.0f;
        // if (params.shouldRejectInnerShift &&
        //    (!shiftPathSample.pathRcInfo.primaryHitInfo.isValid() ||
        //     !isValidGeometry(pathSampleRecord.pathRcInfo.primaryShadingData, shiftPathSample.pathRcInfo.primaryShadingData)))
        // {
        //     shift = float3(0.0f);
        //     jacobian = 0.0f;
        //     shiftPathSample.pathRcInfo = PathRcInfo();
        // }

        float denom = targetFunc(base.xyz) + targetFunc(shift.xyz) * jacobian;
        float w = denom > 0.0f ? targetFunc(base.xyz) / denom : 0.0f;

        float s = (pathSampleRecord.index == 0) ? 1.0f : -1.0f;
        pathSampleRecord.innerJacobian = jacobian;

        return float4(s * w * (shift.xyz * jacobian - base.xyz), 0.0f);
    }

    void computePrimaryHit<RTQuery : IRTQuery>(const CameraData cameraData,
                                               float2 fracPixelLoc,
                                               RTQuery rtq,
                                               inout VBuffer primaryHitInfo)
    {
        Ray cameraRay = computeRayPinhole(cameraData, fracPixelLoc, params.frameDim);
        float primaryHitT;
        float hitT;
        bool isHit = rtq.traceRay(cameraRay, primaryHitInfo.primaryHitInfo, hitT);
        let lod = ExplicitLodTextureSampler(0.0f);

        // shading data
        primaryHitInfo.cameraRay = cameraRay;
        ShadingData primaryShadingData = loadShadingData(primaryHitInfo.primaryHitInfo, cameraRay, lod);
        let mi = gScene.materials.getMaterialInstance(primaryShadingData, lod);
        primaryHitInfo.albedo = mi.getProperties(primaryShadingData).diffuseReflectionAlbedo +
                                mi.getProperties(primaryShadingData).diffuseTransmissionAlbedo;
    }

    void generatePrimayHit<RTQuery : IRTQuery>(const uint2 pixel,
                                               RTQuery rtq,
                                               out VBuffer primaryHitInfo)
    {
        primaryHitInfo.pixel = pixel;
        float2 fracPixelLoc = float2(pixel + subPixelUV[pixel]);
        computePrimaryHit(gScene.camera.data, fracPixelLoc, rtq, primaryHitInfo);
    }

    // Generate paths
    float4 generatePath<RTQuery : IRTQuery>(const uint2 pixel,
                                            const VBuffer primaryHitInfo,
                                            const uint seed,
                                            bool useDefaultSeed,
                                            RTQuery rtq,
                                            inout PathSampleRecord pathSampleRecord)
    {
        SampleGenerator sg = useDefaultSeed ? SampleGenerator(pixel, seed) :
                                              SampleGenerator(pathSampleRecord.initRandomSeed.xy, pathSampleRecord.initRandomSeed.z);
        float4 result = float4(float3(0.0), 1.0);

        // Primary ray
        VBuffer vBuffer = primaryHitInfo;
        float2 fracPixelLoc = subPixelUV[pixel];

        // computePrimaryHit(gScene.camera.data,
        //                   float2(pixel + fracPixelLoc),
        //                   rtq, vBuffer);
        // vBuffer.pixel = pixel;
        Ray cameraRay = vBuffer.cameraRay;
        HitInfo primaryHit = vBuffer.primaryHitInfo;

        if (primaryHit.isValid())
        {
            Ray scatterRay;
            PathVertex path = PathVertex();
            // float pixelWeight = evalGaussian(pixelf - pixelCenter, kPixelGaussianSigma);
            // path.throughput *= pixelWeight / pixelWeight;
            {
                uint INIT_SAMPLE_COUNT = sg.getCount();
                handleHit(cameraRay, primaryHit, emissiveLightSampler, rtq, sg, scatterRay, path);
                uint sampleConsumed = sg.getCount() - INIT_SAMPLE_COUNT;
                consumeRng(sg, kNumOfSamplesPerHit - sampleConsumed);
            }

            Ray ray = scatterRay;
            while (!path.terminated)
            {
                uint INIT_SAMPLE_COUNT = sg.getCount();
                HitInfo hit;
                float hitT;

                bool isHit = rtq.traceRay(ray, hit, hitT);
                Ray scatterRay;
                if (hit.isValid())
                {
                    handleHit(ray, hit, emissiveLightSampler, rtq, sg, scatterRay, path);
                    ray = scatterRay;
                }
                else
                {
                    path.terminated = true;
                }
                uint sampleConsumed = sg.getCount() - INIT_SAMPLE_COUNT;
                consumeRng(sg, kNumOfSamplesPerHit - sampleConsumed);
            }

            path.risState.finalize();
            result = float4(path.risState.integrand, path.risState.W);

            // Save path state
            pathSampleRecord.isLastVertexNEE = path.risState.isLastVertexNEE;
            pathSampleRecord.pathRcInfo = path.risState.pathRcInfo;
        }

        pathSampleRecord.pathRcInfo.fracPixelLoc = fracPixelLoc;
        // Save the path sample record
        if (useDefaultSeed)
        {
            pathSampleRecord.initRandomSeed = uint3(pixel, seed);
        }

        return result;
  }

  float4 randomPlayPath<RTQuery : IRTQuery>(const uint2 pixel,
                                            const VBuffer primaryHitInfo,
                                            RTQuery rtq,
                                            inout PathSampleRecord pathSample)
  {
        float4 result = float4(0.0f);

        SampleGenerator sg = SampleGenerator(pathSample.initRandomSeed.xy, pathSample.initRandomSeed.z);

        // Primary ray
        // float2 pixelCenter = pixel + float2(0.5f);
        // float2 jitter2D = sampleGaussian(sampleNext2D(sg), kPixelGaussianSigma);
        // float2 pixelf = pixelCenter + jitter2D;
        // float3 cameraRayPos = gScene.camera.getPosition();
        // float3 cameraRayDir = computeCameraRayDirection(pixelf, params.frameDim);

        // Ray cameraRay = Ray(cameraRayPos, cameraRayDir, 0.0f, kRayTMax);
        Ray cameraRay = primaryHitInfo.cameraRay;
        HitInfo primaryHit = primaryHitInfo.primaryHitInfo;

        if (primaryHit.isValid())
        {
            Ray scatterRay;
            PathVertex path = PathVertex();
            // float pixelWeight = evalGaussian(pixelf - pixelCenter, kPixelGaussianSigma);
            // path.throughput *= pixelWeight / pixelWeight;
            {
                uint INIT_SAMPLE_COUNT = sg.getCount();
                handleHitReplay(cameraRay,
                            primaryHit,
                            emissiveLightSampler,
                            pathSample,
                            rtq,
                            sg,
                            scatterRay,
                            path);
                uint sampleConsumed = sg.getCount() - INIT_SAMPLE_COUNT;
                consumeRng(sg, kNumOfSamplesPerHit - sampleConsumed);
            }

            Ray ray = scatterRay;
            while (!path.terminated)
            {
                HitInfo hit;
                float hitT;
                bool isHit = rtq.traceRay(ray, hit, hitT);
                Ray scatterRay;
                if (hit.isValid())
                {
                    uint INIT_SAMPLE_COUNT = sg.getCount();
                    handleHitReplay(ray,
                                    hit,
                                    emissiveLightSampler,
                                    pathSample,
                                    rtq,
                                    sg,
                                    scatterRay,
                                    path);
                    uint sampleConsumed = sg.getCount() - INIT_SAMPLE_COUNT;
                    consumeRng(sg, kNumOfSamplesPerHit - sampleConsumed);
                    ray = scatterRay;
                }
                else
                {
                    path.terminated = true;
                }
            }
            result.xyz = path.radiance;
        }


        return result;
  }

    void handleHit<RTQuery : IRTQuery>(const Ray ray,
                                       const HitInfo hit,
                                       EmissiveLightSampler emissiveLightSampler,
                                       RTQuery rtq,
                                       inout SampleGenerator sg,
                                       out Ray scatterRay,
                                       inout PathVertex path)
    {
        scatterRay = {};
        const bool isCurveHit = hit.getType() == HitType::Curve;
        let lod = ExplicitLodTextureSampler(0.0f);

        // load shading data
        ShadingData sd = loadShadingData(hit, ray, lod);

        // create material instance
        let mi = gScene.materials.getMaterialInstance(sd, lod);
        let isPrimaryHit = (path.length == 0);
        let isTriangleHit = (hit.getType() == HitType::Triangle);

        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;
        bool applyNEE = kUseNEE & hasNonDeltaLobes;

        // Compute the last vertex states for hybrid shift
        float distanceToPrevVertex = length(sd.posW - ray.origin);
        float geometryFactor = abs(dot(normalize(sd.faceN), normalize(sd.V))) / sqr(distanceToPrevVertex);
        bool isDistanceFar = distanceToPrevVertex > params.distanceThreshold * params.sceneRadius;
        BSDFProperties bsdfProperties = mi.getProperties(sd);
        bool isCurrentVertexClassifiedAsRoughForNEE = false;
        {
              const bool hasDiffuse = (lobeTypes & (uint)LobeType::Diffuse) != 0;
              const bool highRoughness = (lobeTypes & (uint)LobeType::Specular) != 0 &&
                                          bsdfProperties.roughness > params.roughnessThreshold;
              isCurrentVertexClassifiedAsRoughForNEE = hasDiffuse || highRoughness;
        }
        path.lastVertexState.Init( isDistanceFar, path.lastBSDFComponent, path.isDelta(),
                                  path.isTransmission(),
                                  path.isRough(),
                                  isCurrentVertexClassifiedAsRoughForNEE);

        // emitted light
        bool isBSDFSelected = false;
        if ((!applyNEE || isPrimaryHit || !isTriangleHit))
        {
            float3 integrandBSDF = path.throughput * mi.getProperties(sd).emission;
            if (!isPrimaryHit)
            {
                // isBSDFSelected = path.addBSDFVertex(integrandBSDF, 1.0f, sg);

                if (isBSDFSelected && path.pathRcInfo.rcHitIndex != NO_RCINDEX)
                {
                      path.risState.pathRcInfo = path.pathRcInfo;
                      path.risState.pathRcInfo.isLastVertexNEE = false;
                      path.risState.pathRcInfo.pathLength = path.length;
                }
            }
        }
        else
        {
            float mis_weight = 1.0f;
            float bsdfSamplingPdf = path.lastBsdfSamplingpdf;
            float lightSamplingPdf = 1.0f;

            if (!path.isDelta())
            {
                // prepare light sampling pdf
                TriangleHit triHit = hit.getTriangleHit();
                TriangleLightHit lightHit;
                lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(triHit.instanceID,
                                                                                 triHit.primitiveIndex);
                lightHit.posW = sd.posW;
                lightHit.normalW = sd.getOrientedFaceNormal();

                bool upperHemisphere = path.sampleUpperHemisphere;
                lightSamplingPdf = emissiveLightSampler.evalPdf(ray.origin,
                                                                path.lastNormal,
                                                                upperHemisphere,
                                                                lightHit);

                mis_weight = balanceHeuristic(1, path.pdf, 1, lightSamplingPdf);
            }

            float3 integrand = mis_weight * path.throughput * mi.getProperties(sd).emission;
            isBSDFSelected = path.addBSDFVertex(integrand, 1.0f, sg);
            {
                // Check if the current vertex can be connected
                bool canConnect = path.length >= 1u && isDistanceFar && path.lastVertexState.isLastVertexRough;

                // If we have got a reconnection vertex earlier
                if (isBSDFSelected && path.pathRcInfo.rcHitIndex != NO_RCINDEX)
                {
                      path.risState.pathRcInfo = path.pathRcInfo;
                      path.risState.pathRcInfo.isLastVertexNEE = false;
                      path.risState.pathRcInfo.pathLength = path.length;
                      path.risState.pathRcInfo.emission = path.pathRcInfo.emission * mis_weight * mi.getProperties(sd).emission;
                }
                else if (isBSDFSelected && canConnect && path.pathRcInfo.rcHitIndex == NO_RCINDEX)
                {
                      // If we do not find a reconnection vertex
                      path.risState.pathRcInfo = PathRcInfo();
                      path.risState.pathRcInfo.rcHit = hit;
                      path.risState.pathRcInfo.rcHitIndex = path.length;
                      path.risState.pathRcInfo.rcJacobian = geometryFactor * bsdfSamplingPdf;
                      path.risState.pathRcInfo.lightPdf = lightSamplingPdf;
                      path.risState.pathRcInfo.geometryTerm = geometryFactor;
                      path.risState.pathRcInfo.pathLength = path.length;
                      path.risState.pathRcInfo.isLastVertexNEE = false;
                      path.risState.pathRcInfo.emission = bsdfProperties.emission;
                      path.risState.pathRcInfo.vertexKMinusOneBsdfComponent = path.lastBSDFComponent;
                      path.risState.pathRcInfo.vertexKMinusOnePathFlags = path.flags;
                      path.risState.pathRcInfo.vertexKPathFlags = 0u;
                      setFlag(path.risState.pathRcInfo.vertexKPathFlags, PathFlags::rough, true);
                      path.risState.pathRcInfo.isDeltaOnlyPath = false;
                }
                else if (isBSDFSelected && !canConnect && path.pathRcInfo.rcHitIndex == NO_RCINDEX)
                {
                      // If current path is selected but no reconnection vertex is found.
                      path.risState.pathRcInfo = PathRcInfo();
                      path.risState.pathRcInfo.pathLength = path.length;
                      path.risState.pathRcInfo.isLastVertexNEE = false;
                      path.risState.pathRcInfo.emission = float3(0.0f);
                      path.risState.pathRcInfo.isDeltaOnlyPath = path.pathRcInfo.isDeltaOnlyPath;
                }
            }
        }

        bool isLight = any(mi.getProperties(sd).emission > float3(0.0f));
        if (path.length >= kMaxBounces || isLight)
        {
            path.terminated = true;
        }

        if (!path.terminated && applyNEE)
        {
            uint INIT_SAMPLE_COUNT = sg.getCount();
            LightSample ls;
            HitInfo neeHitInfo;

            bool sampleUpper = isCurveHit || ((lobeTypes & (uint)LobeType::NonDeltaReflection) != 0);
            bool sampleLower = isCurveHit || ((lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0);
            bool sampleUpperHemisphere = sampleUpper && !sampleLower;

            bool sampleEmissive = sampleEmissive(sd, emissiveLightSampler, sampleUpperHemisphere, sg, ls, neeHitInfo);

            bool canCurVetexConnect = path.length >= 1u && isDistanceFar && path.lastVertexState.isLastVertexRough &&
                                      path.lastVertexState.isVertexClassifiedAsRoughForNEE;

            float distanceToLight = length(sd.posW - ls.lightPos);
            bool isFarFromLight = distanceToLight > params.distanceThreshold * params.sceneRadius;
            bool canNEEConnect = path.length >= 0u && isFarFromLight &&
                                 path.lastVertexState.isVertexClassifiedAsRoughForNEE;
            if (sampleEmissive)
            {
                float3 weight;
                float3 integrandNEE = evalEmissive(sd, ls, mi, rtq, weight, sg, path);
                bool isSelected = path.addNEEVertex(integrandNEE, 1.0f, sg);
                // If we need to reconnect in NEE vertex, we will
                // mark it directly.
                if (isSelected && path.pathRcInfo.rcHitIndex == NO_RCINDEX)
                {
                    // 2 cases
                    // 1. Vertex K is an NEE vertex
                    // 2. Vertex K + 1 is an NEE vertex
                    if (!canCurVetexConnect && canNEEConnect)
                    {
                        path.risState.pathRcInfo = PathRcInfo();
                        path.risState.pathRcInfo.rcHit = neeHitInfo;
                        path.risState.pathRcInfo.rcJacobian = ls.geometryTerm * ls.pdf;
                        path.risState.pathRcInfo.rcHitIndex = path.length + 1;
                        path.risState.pathRcInfo.isLastVertexNEE = true;
                        path.risState.pathRcInfo.lightPdf = ls.pdf;
                        path.risState.pathRcInfo.geometryTerm = ls.geometryTerm;
                        path.risState.pathRcInfo.isLastVertexNEE = true;
                        path.risState.pathRcInfo.pathLength = path.length + 1;
                        path.risState.pathRcInfo.emission = ls.Le;
                        path.risState.pathRcInfo.vertexKMinusOneBsdfComponent = 0xf;
                        path.risState.pathRcInfo.vertexKMinusOnePathFlags = 0u;
                        path.risState.pathRcInfo.vertexKPathFlags = 0u;
                        setFlag(path.risState.pathRcInfo.vertexKPathFlags, PathFlags::rough, true);
                        setFlag(path.risState.pathRcInfo.vertexKMinusOnePathFlags, PathFlags::rough, true);
                        path.risState.pathRcInfo.isDeltaOnlyPath = false;
                    }
                    else if (canCurVetexConnect)
                    {
                        path.risState.pathRcInfo = PathRcInfo();
                        path.risState.pathRcInfo.rcHit = hit;
                        path.risState.pathRcInfo.rcJacobian = geometryFactor * ls.pdf * path.lastBsdfSamplingpdf;
                        path.risState.pathRcInfo.rcHitIndex = path.length;
                        path.risState.pathRcInfo.isLastVertexNEE = true;
                        path.risState.pathRcInfo.lightPdf = ls.pdf;
                        path.risState.pathRcInfo.geometryTerm = geometryFactor;
                        path.risState.pathRcInfo.pathLength = path.length + 1;
                        path.risState.pathRcInfo.emission = ls.Le;
                        path.risState.pathRcInfo.rcWi = normalize(ls.lightPos - ls.visOrigin);
                        path.risState.pathRcInfo.vertexKMinusOneBsdfComponent = path.lastBSDFComponent;
                        path.risState.pathRcInfo.vertexKBsdfComponent = 0xf;
                        path.risState.pathRcInfo.vertexKMinusOnePathFlags = path.flags;
                        path.risState.pathRcInfo.vertexKPathFlags = 0u;
                        setFlag(path.risState.pathRcInfo.vertexKPathFlags, PathFlags::rough, true);
                        path.risState.pathRcInfo.isDeltaOnlyPath = false;
                    }
                    else
                    {
                        path.risState.pathRcInfo = PathRcInfo();
                        path.risState.pathRcInfo.isLastVertexNEE = true;
                        path.risState.pathRcInfo.pathLength = path.length + 1;
                        path.risState.pathRcInfo.emission = ls.Le;
                        path.risState.pathRcInfo.isDeltaOnlyPath = false;
                    }
                }
                else if (isSelected && path.pathRcInfo.rcHitIndex != NO_RCINDEX)
                {
                    // If we have got a reconnection index
                    path.risState.pathRcInfo = path.pathRcInfo;
                    path.risState.pathRcInfo.pathLength = path.length + 1;
                    path.risState.pathRcInfo.isLastVertexNEE = true;
                    path.risState.pathRcInfo.isDeltaOnlyPath = false;
                    path.risState.pathRcInfo.emission = path.pathRcInfo.emission * weight;
                }
            }
            uint sampleConsumed = sg.getCount() - INIT_SAMPLE_COUNT;
            consumeRng(sg, kNumOfSamplesPerNEEHit - sampleConsumed);
        }

        // Compute scatter ray
        if (!path.terminated)
        {
            float lastScatterPdf = path.lastBsdfSamplingpdf;
            uint lastBSDFComponent = path.lastBSDFComponent;
            uint lastPathFlags = path.flags;
            bool isValid = computeScatterRay(sd, mi, sg,
                                             isCurveHit,
                                             scatterRay,
                                             path);
            float curScatterPdf = path.lastBsdfSamplingpdf;
            uint curBSDFComponent = path.lastBSDFComponent;
            uint curPathFlags = path.flags;

            bool isRough = path.lastVertexState.isLastVertexRough && path.isRough();
            bool canConnect = path.length >= 1u && path.lastVertexState.isCurrentVertexFarFromPrev && isRough;
            // If we find a reconnection vertex
            if (isValid && path.pathRcInfo.rcHitIndex == NO_RCINDEX && canConnect)
            {
                path.pathRcInfo.rcHit = hit;
                path.pathRcInfo.rcHitIndex = path.length;
                path.pathRcInfo.rcJacobian = lastScatterPdf * geometryFactor * curScatterPdf;
                path.pathRcInfo.lightPdf = 1.0f; // Actually useless
                path.pathRcInfo.geometryTerm = geometryFactor;
                path.pathRcInfo.rcWi = normalize(scatterRay.dir);
                path.pathRcInfo.vertexKMinusOneBsdfComponent = lastBSDFComponent;
                path.pathRcInfo.vertexKMinusOnePathFlags = lastPathFlags;
                path.pathRcInfo.vertexKBsdfComponent = curBSDFComponent;
                path.pathRcInfo.vertexKPathFlags = curPathFlags;
                path.pathRcInfo.vertexKBsdfLobes = path.lastLobeType;
                path.pathRcInfo.vertexKSampleUpperHemisphere = path.sampleUpperHemisphere;
                path.pathRcInfo.emission = float3(1.0f);
            }

            if (!isValid)
            {
                path.terminated = true;
            }
        }

        path.length++;
    }

    void handleHitReplay<RTQuery : IRTQuery>(const Ray ray,
                                             const HitInfo hit,
                                             EmissiveLightSampler emissiveLightSampler,
                                             const PathSampleRecord pathSample,
                                             RTQuery rtq,
                                             inout SampleGenerator sg,
                                             out Ray scatterRay,
                                             inout PathVertex path)
    {
        scatterRay = {};
        const bool isCurveHit = hit.getType() == HitType::Curve;
        let lod = ExplicitLodTextureSampler(0.0f);

        // load shading data
        ShadingData sd = loadShadingData(hit, ray, lod);

        // create material instance
        let mi = gScene.materials.getMaterialInstance(sd, lod);
        let isPrimaryHit = (path.length == 0);
        let isTriangleHit = (hit.getType() == HitType::Triangle);

        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;
        bool applyNEE = kUseNEE & hasNonDeltaLobes;

        // Compute last vertex states for hybrid shift
        float distanceToPrevVertex = length(sd.posW - ray.origin);
        float geometryFactor = abs(dot(sd.faceN, sd.V)) / sqr(distanceToPrevVertex);
        bool isDistanceFar = distanceToPrevVertex > params.distanceThreshold * params.sceneRadius;
        BSDFProperties bsdfProperties = mi.getProperties(sd);
        bool isVertexClassifiedAsRoughForNEE = false;
        {
              const bool hasDiffuse = (lobeTypes & (uint)LobeType::Diffuse) != 0u;
              const bool highRoughness = (lobeTypes & (uint)LobeType::Specular) != 0 &&
                                          bsdfProperties.roughness > params.roughnessThreshold;
              isVertexClassifiedAsRoughForNEE = hasDiffuse || highRoughness;
        }
        path.lastVertexState.Init( isDistanceFar, path.lastBSDFComponent, path.isDelta(),
                                  path.isTransmission(),
                                  path.isRough(),
                                  isVertexClassifiedAsRoughForNEE );

        // emitted light
        if ((!pathSample.pathRcInfo.isLastVertexNEE && path.length == pathSample.pathRcInfo.pathLength))
        {
            float3 integrandBSDF;
            if (!applyNEE || isPrimaryHit || !isTriangleHit)
            {
                integrandBSDF = path.throughput * mi.getProperties(sd).emission;
                // path.addBSDFVertex(integrandBSDF, 1.0f, sg);
            }
            else
            {
                float lightSamplingPdf = 1.0f;
                float mis_weight = 1.0f;

                if (!path.isDelta())
                {
                    // prepare light sampling pdf
                    TriangleHit triHit = hit.getTriangleHit();
                    TriangleLightHit lightHit;
                    lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(triHit.instanceID,
                                                                                     triHit.primitiveIndex);
                    lightHit.posW = sd.posW;
                    lightHit.normalW = sd.getOrientedFaceNormal();

                    bool upperHemisphere = path.sampleUpperHemisphere;
                    lightSamplingPdf = emissiveLightSampler.evalPdf(ray.origin,
                                                                    path.lastNormal,
                                                                    upperHemisphere,
                                                                    lightHit);

                    mis_weight = balanceHeuristic(1, path.pdf, 1, lightSamplingPdf);
                }

                integrandBSDF = mis_weight * path.throughput * mi.getProperties(sd).emission;
            }

            bool canConnect = path.length >= 1u && isDistanceFar && path.lastVertexState.isLastVertexRough;
            if (path.pathRcInfo.rcHitIndex == NO_RCINDEX && canConnect)
            {
                path.pathRcInfo.rcHitIndex = path.length;
                path.pathRcInfo.isDeltaOnlyPath = false;
            }
            path.pathRcInfo.isLastVertexNEE = false;
            path.pathRcInfo.pathLength = path.length;

            path.radiance = integrandBSDF;
            path.terminated = true;
        }

        if (!isPrimaryHit && isTriangleHit && applyNEE)
        {
            consumeRng(sg, 1u);
        }

        bool isLight = any(mi.getProperties(sd).emission > 0.0f);
        if (path.length >= kMaxBounces || isLight)
        {
            path.terminated = true;
        }

        if (pathSample.pathRcInfo.isLastVertexNEE && pathSample.pathRcInfo.pathLength - 1 == path.length)
        {
            LightSample ls;
            HitInfo neeHitInfo;

            bool sampleUpper = isCurveHit || ((lobeTypes & (uint)LobeType::NonDeltaReflection) != 0);
            bool sampleLower = isCurveHit || ((lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0);
            bool sampleUpperHemisphere = sampleUpper && !sampleLower;

            bool sampleEmissive = sampleEmissive(sd, emissiveLightSampler, sampleUpperHemisphere, sg, ls, neeHitInfo);

            path.pathRcInfo.isLastVertexNEE = true;
            path.pathRcInfo.pathLength = path.length + 1;
            path.pathRcInfo.isDeltaOnlyPath = false;

            if (sampleEmissive)
            {
                float3 weight;
                float3 integrandNEE = evalEmissive(sd, ls, mi, rtq, weight, sg, path);
                path.radiance = integrandNEE;
            }

            path.terminated = true;
        }
        else if (!path.terminated && applyNEE)
        {
            // consume the number of random numbers in an NEE hit.
            consumeRng(sg, kNumOfSamplesPerNEEHit);
        }

        // Compute scatter ray
        if (!path.terminated)
        {
            // float lastScatterPdf = path.lastBsdfSamplingpdf;
            bool isValid = computeScatterRay(sd, mi, sg,
                                             isCurveHit,
                                             scatterRay,
                                             path);

            bool isPathRough = path.lastVertexState.isLastVertexRough && path.isRough();
            bool canConnect = path.length >= 1u && path.lastVertexState.isCurrentVertexFarFromPrev && isPathRough;
            if (isValid && path.pathRcInfo.rcHitIndex == NO_RCINDEX && canConnect)
            {
                path.pathRcInfo.rcHitIndex = path.length;
            }
            if (!isValid)
            {
                print("Warning: Cannot have scatter ray.", isValid);
                path.terminated = true;
            }
        }
        path.length++;
    }

    bool computeScatterRay(ShadingData sd,
                           const IMaterialInstance mi,
                           inout SampleGenerator sg,
                           bool isCurveHit,
                           out Ray ray,
                           inout PathVertex path)
    {
        ray = {};
        BSDFSample bsdfSample;
        sd.mtl.setLobeMask(1);
        if (mi.sample(sd, sg, bsdfSample, true))
        {
            path.lastFlags = path.flags;
            path.clearFlags();

            float3 rayOrigin = computeRayOrigin(sd.posW, sd.getOrientedFaceNormal());
            if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission))
            {
                rayOrigin = sd.computeRayOrigin(false);
                path.setTransmission();
            }
            float3 rayDir = bsdfSample.wo;
            float3 weight = bsdfSample.weight;

            float roughness = mi.getProperties(sd).roughness;
            bool isDiffuse = bsdfSample.isLobe(LobeType::Diffuse) || (bsdfSample.isLobe(LobeType::Specular) &&
                             roughness > params.roughnessThreshold);
            if (isDiffuse)
            {
                path.setRough();
            }
            else
            {
                path.setSpecular();
            }

            if (bsdfSample.isLobe(LobeType::Delta))
            {
                path.setDelta();
            }
            path.pathRcInfo.isDeltaOnlyPath &= path.isDelta();

            uint lobeTypes = bsdfSample.lobeType;
            bool sampleUpper = isCurveHit || ((lobeTypes & (uint)LobeType::NonDeltaReflection) != 0);
            bool sampleLower = isCurveHit || ((lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0);
            path.sampleUpperHemisphere = sampleUpper && !sampleLower;

            path.throughput *= weight;
            path.pathRcInfo.emission *= weight;
            path.lastBsdfSamplingpdf = bsdfSample.pdf; // Store BSDF pdf for MIS.
            path.lastNormal = sd.getOrientedFaceNormal();
            path.lastLobeType = bsdfSample.lobeType;
            path.lastBSDFComponent = bsdfSample.getComponentType();
            sd.mtl.setLobeMask(~(1 << path.lastBSDFComponent));
            path.pdf = bsdfSample.pdf + mi.evalPdf(sd, bsdfSample.wo);
            ray = Ray(rayOrigin, rayDir, 0.0f, kRayTMax);

            return any(path.throughput > 0.0f);
        }

        return false;
    }

    float3 evalEmissive<RTQuery : IRTQuery>(ShadingData sd,
                                            const LightSample ls,
                                            IMaterialInstance mi,
                                            RTQuery rtq,
                                            out float3 weight,
                                            inout SampleGenerator sg,
                                            inout PathVertex path)
    {
        float3 wo = normalize(ls.lightPos - ls.visOrigin);

        Ray shadowRay = ls.getVisibilityRay();
        bool visible = rtq.traceVisibilityRay(shadowRay);

        // float dis = length(ls.lightPos - sd.posW);
        // float3 bsdfValue = mi.eval(sd, wo, sg);
        // float bsdfSamplingPdf = mi.evalPdf(sd, wo);

        float3 bsdfValue = mi.eval(sd, wo, sg);
        float bsdfPdf = mi.evalPdf(sd, wo, true);

        float misWeight = balanceHeuristic(1, ls.pdf, 1, bsdfPdf);

        weight = visible ? misWeight * ls.Li * bsdfValue : float3(0.0f);

        return path.throughput * weight;
    }

    bool computeReconnectionThroughputAndJacobian<RTQuery : IRTQuery>( RTQuery rtq,
                                                                       inout PathRcInfo pathRcInfo,
                                                                       ReconnectionData reconnectionData,
                                                                       inout PathVertex path,
                                                                       out float3 reconnectionThroughput,
                                                                       out float reconnectionJacobian,
                                                                       out float reconnectionRcJacobian )
    {
        SampleGenerator sg = SampleGenerator(uint2(1), 1);

        bool isVertexKLight = reconnectionData.isVertexKLight;
        bool isVertexKNEE   = isVertexKLight && pathRcInfo.isLastVertexNEE;

        bool isVertexKConnectToNEE = pathRcInfo.rcHitIndex + 1 == pathRcInfo.pathLength && pathRcInfo.isLastVertexNEE;

        // TODO
        ShadingData dstVertexKMinusOneSd = loadShadingData(reconnectionData.dstVertexKMinusOne,
                                                        reconnectionData.dstVertexKMinusOneRay,
                                                        ExplicitLodTextureSampler(0.0f));
        bool isTransmissionKMinusOne = isTransmission(pathRcInfo.vertexKMinusOnePathFlags);
        const IMaterialInstance dstVertexKMinusOneMI = gScene.materials.getMaterialInstance(dstVertexKMinusOneSd,
                                                    ExplicitLodTextureSampler(0.0f));
        float3 dstVertexKMinusOnePosW = dstVertexKMinusOneSd.computeRayOrigin(!isTransmissionKMinusOne);

        ShadingData dstVertexKSd = loadShadingData(reconnectionData.dstVertexK,
                                                    Ray(dstVertexKMinusOnePosW, float3(0.0f)),
                                                    ExplicitLodTextureSampler(0.0f));
        const IMaterialInstance dstVertexKMI = gScene.materials.getMaterialInstance(dstVertexKSd, ExplicitLodTextureSampler(0.0f));
        float3 dstVertexKPosW = dstVertexKSd.posW;
        float3 dstVertexKN = dstVertexKSd.getOrientedFaceNormal();
        float3 dirKMinusOneToK = normalize(dstVertexKPosW - dstVertexKMinusOnePosW);
        float distKMinusOneToK = sqr(length(dstVertexKPosW - dstVertexKMinusOnePosW));
        float cosineKMinusOneToK = abs(dot(dirKMinusOneToK, dstVertexKN));
        bool dstKFrontFacing = dstVertexKSd.frontFacing;

        float geometryTerm = reconnectionData.isDstKExists ? cosineKMinusOneToK / distKMinusOneToK : 1.0f;
        float dstKMinusOnePdfBSDF, dstKMinusOnePdfBSDFForMIS;
        float3 bsdfKMinusOne = dstVertexKMinusOneMI.evalBsdfAndPdf( dstVertexKMinusOneSd, dirKMinusOneToK,
                                    sg, isVertexKNEE ? 0xf : 1 << reconnectionData.bsdfComponentKMinusOne,
                                    dstKMinusOnePdfBSDF, dstKMinusOnePdfBSDFForMIS );

        float lightPdf1 = geometryTerm > 0.0f ? pathRcInfo.lightPdf * pathRcInfo.geometryTerm / geometryTerm : 0.0f;
        float dstPdf1 = isVertexKNEE ? lightPdf1 : dstKMinusOnePdfBSDF;

        bool isDstKMinusOneLight = any(dstVertexKMinusOneMI.getProperties(dstVertexKMinusOneSd).emission > 0.0f);
        if (isDelta(pathRcInfo.vertexKMinusOnePathFlags) || isDelta(pathRcInfo.vertexKPathFlags) || isDstKMinusOneLight)
        {
            reconnectionJacobian = 0.0f;
            reconnectionRcJacobian = 0.0f;
            reconnectionThroughput = float3(0.0f);
            return false;
        }

        reconnectionThroughput = pathRcInfo.emission;
        reconnectionRcJacobian = reconnectionData.isDstKExists ? geometryTerm : 1.0f;
        reconnectionThroughput *= (dstPdf1 > 0.0f ? bsdfKMinusOne / dstPdf1 : 0.0f);
        reconnectionRcJacobian *= (reconnectionData.isDstKExists ? dstPdf1 : 1.0f);
        if (reconnectionData.isDstKExists && reconnectionData.isVertexKLight && !dstKFrontFacing)
        {
            reconnectionThroughput = float3(0.0f);
            reconnectionJacobian = 0.0f;
            reconnectionRcJacobian = 0.0f;
            return false;
        }
        float misKMinusOne = kUseNEE && reconnectionData.isVertexKLight ? balanceHeuristic( 1, isVertexKNEE ? lightPdf1 : dstKMinusOnePdfBSDFForMIS,
                                                                                            1, isVertexKNEE ? dstKMinusOnePdfBSDFForMIS : lightPdf1 ) : 1.0f;
        reconnectionThroughput *= misKMinusOne;

        bool needSecondPart = reconnectionData.isDstKExists && !isVertexKLight;
        float dstKPdfBSDF, dstKPdfBSDFForMIS;
        float3 bsdfK = needSecondPart ? dstVertexKMI.evalBsdfAndPdf( dstVertexKSd, reconnectionData.vertexKWi,
            sg, isVertexKConnectToNEE ? 0xf : 1 << reconnectionData.bsdfComponentK,
            dstKPdfBSDF, dstKPdfBSDFForMIS ) : float3(1.0f);
        float lightPdf2 = pathRcInfo.lightPdf;
        float dstPdf2 = isVertexKConnectToNEE ? lightPdf2 : dstKPdfBSDF;
        float misK = (kUseNEE && isVertexKConnectToNEE) ? balanceHeuristic( 1, isVertexKConnectToNEE ? lightPdf2 : dstKPdfBSDFForMIS,
                                                                          1, isVertexKConnectToNEE ? dstKPdfBSDFForMIS : lightPdf2 ) : 1.0f;
        reconnectionThroughput *= misK;
        reconnectionThroughput *= (needSecondPart ? (bsdfK / dstPdf2) : 1.0f);
        reconnectionRcJacobian *= (needSecondPart ? dstPdf2 : 1.0f);
        reconnectionJacobian = pathRcInfo.rcJacobian > 0.0f ? reconnectionRcJacobian / pathRcInfo.rcJacobian : 0.0f;

        path.lastBsdfSamplingpdf = dstKPdfBSDF;
        path.pdf = dstKPdfBSDFForMIS;
        path.lastBSDFComponent = reconnectionData.bsdfComponentK;
        path.lastNormal = dstVertexKSd.getOrientedFaceNormal();
        path.flags = pathRcInfo.vertexKPathFlags;
        path.lastLobeType = pathRcInfo.vertexKBsdfLobes;
        path.sampleUpperHemisphere = pathRcInfo.vertexKSampleUpperHemisphere;

        bool isVisible = isVertexKNEE ? evalSegmentVisibility(false, dstVertexKMinusOnePosW, computeRayOrigin(dstVertexKPosW, dstVertexKN), rtq, false) :
                                        evalSegmentVisibility(false, dstVertexKMinusOnePosW, dstVertexKPosW, rtq, false);
        if (!reconnectionData.isDstKExists)
        {
            isVisible = false;
        }
        bool isValidShift = !isJacobianInvalid(reconnectionJacobian) && !isIntegrandInvalid(reconnectionThroughput) && isVisible;
        // if (!isValidShift)
        // {
        //     print("reconnectionJacobian:", reconnectionJacobian);
        //     print("dstPdf1:", dstPdf1);
        //     print("emission:", pathRcInfo.emission);
        //     print("lightPdf1:", lightPdf1);
        //     print("bsdfKMinusOnePdf:", dstKMinusOnePdfBSDF);
        //     print("bsdfKMinusOnePdfForMIS", dstKMinusOnePdfBSDFForMIS);
        //     print("misMinusOne:", misKMinusOne);
        //     print("needSecondPart:", needSecondPart);
        //     print("dstPdf2:", dstPdf2);
        //     print("bsdfK:", bsdfK);
        //     print("reconnectionThroughput:", reconnectionThroughput);
        //     print("isVisible:", isVisible);
        // }
        reconnectionJacobian = isValidShift ? reconnectionJacobian : 0.0f;
        reconnectionRcJacobian = isValidShift ? reconnectionRcJacobian : 0.0f;
        reconnectionThroughput = isValidShift ? reconnectionThroughput : float3(0.0f);
        pathRcInfo.geometryTerm = geometryTerm;
        pathRcInfo.rcJacobian = reconnectionRcJacobian;
        if (isVertexKLight)
        {
            pathRcInfo.lightPdf = lightPdf1;
        }
        else
        {
            pathRcInfo.lightPdf = lightPdf2;
        }

        path.length = (uint16_t) pathRcInfo.rcHitIndex;
        if (pathRcInfo.pathLength > pathRcInfo.rcHitIndex && !isVertexKConnectToNEE)
        {
            path.length++;
        }

        float distanceFromKMinusOneToK = length(dstVertexKMinusOnePosW - dstVertexKSd.posW);
        bool isDistanceFar = distanceFromKMinusOneToK > params.sceneRadius * params.distanceThreshold;

        uint dstVertexKMinusOneLobeTypes = dstVertexKMinusOneMI.getLobeTypes(dstVertexKMinusOneSd);
        bool isDstVertexKMinusOneRough = false;
        {
            bool hasDiffuse = (dstVertexKMinusOneLobeTypes & (uint)LobeType::Diffuse) > 0u;
            bool highRoughness = dstVertexKMinusOneMI.getProperties(dstVertexKMinusOneSd).roughness > params.roughnessThreshold;
            isDstVertexKMinusOneRough = hasDiffuse || highRoughness;
        }
        bool canConnect = isValidShift && isDistanceFar && isDstVertexKMinusOneRough;
        // if (!canConnect)
        // {
        //     print("isDistanceFar:", isDistanceFar);
        //     print("isKRough:", isDstVertexKMinusOneRough);
        //     print("isKMinusOneRough:", isDstVertexKMinusOneRough);
        // }
        bool isPathEnded = pathRcInfo.pathLength <= pathRcInfo.rcHitIndex;
        if (isPathEnded)
        {
            path.pathRcInfo.pathLength = path.length;
            path.terminated = true;

            if (isVertexKNEE)
            {
                path.pathRcInfo.isLastVertexNEE = true;
            }
            else
            {
                path.pathRcInfo.isLastVertexNEE = false;
            }
        }
        else if (isVertexKConnectToNEE)
        {
            path.pathRcInfo.pathLength = path.length + 1;
            path.pathRcInfo.isLastVertexNEE = true;
            path.terminated = true;
        }

        return canConnect;
    }

    float4 areaReSTIRShiftMapping<RTQuery : IRTQuery>( const PathSampleRecord originPath,
                                                       const float2 fracPixelLoc,
                                                       bool isPrevFrame,
                                                       RTQuery rtq,
                                                       out PathSampleRecord shiftedPath,
                                                       out float jacobian )
    {
        CameraData cameraData = isPrevFrame ? prevCameraData : gScene.camera.data;

        // For now, we should have a same fractional pixel location.
        // shiftedPath.pathRcInfo.fracPixelLoc = fracPixelLoc;

        VBuffer shiftPrimaryHitInfo;
        shiftPrimaryHitInfo.pixel = uint2(fracPixelLoc);
        computePrimaryHit(cameraData,
                          fracPixelLoc + originPath.pathRcInfo.fracPixelLoc,
                          rtq,
                          shiftPrimaryHitInfo);
        float4 L = float4(0.0f);
        if (shiftPrimaryHitInfo.primaryHitInfo.isValid())
        {
            L = hybridShiftMapping(originPath, shiftPrimaryHitInfo.pixel,
                                    shiftPrimaryHitInfo, rtq, shiftedPath, jacobian);

            shiftedPath.pathRcInfo.fracPixelLoc = originPath.pathRcInfo.fracPixelLoc;
        }

        return L;
    }

    float4 hybridShiftMapping<RTQuery : IRTQuery>(const PathSampleRecord originPath,
                                                  uint2 pixel,
                                                  VBuffer primaryHitInfo,
                                                  RTQuery rtq,
                                                  out PathSampleRecord shiftedPath,
                                                  out float jacobian)
    {
        float4 result = float4(0.0f);
        shiftedPath.update(originPath);

        SampleGenerator sg = SampleGenerator(originPath.initRandomSeed.xy, originPath.initRandomSeed.z);
        Ray ray = primaryHitInfo.cameraRay;
        HitInfo hit = primaryHitInfo.primaryHitInfo;
        PathVertex path = PathVertex();
        bool shiftSucceed = true;
        {
            Ray scatterRay;

            int rpLength = originPath.pathRcInfo.rcHitIndex == NO_RCINDEX ?
                           originPath.pathRcInfo.pathLength + 1 :
                           originPath.pathRcInfo.rcHitIndex - 1;

            for (int i = 0; i < rpLength; ++i)
            {
                uint INIT_SAMPLE_COUNT = sg.getCount();

                if (hit.isValid())
                {
                    handleHitReplay(ray,
                                    hit,
                                    emissiveLightSampler,
                                    shiftedPath,
                                    rtq,
                                    sg, scatterRay, path);
                    ray = scatterRay;
                }
                else
                {
                    path.terminated = true;
                    break;
                }
                uint sampleConsumed = sg.getCount() - INIT_SAMPLE_COUNT;
                consumeRng(sg, kNumOfSamplesPerHit - sampleConsumed);
                HitInfo curHit;
                float hitT;
                bool isHit = rtq.traceRay(ray, curHit, hitT);
                hit = curHit;
            }

            if (originPath.pathRcInfo.rcHitIndex == NO_RCINDEX)
            {
                shiftSucceed = !originPath.pathRcInfo.isDeltaOnlyPath &&
                               any(path.radiance > 0.0f) &&
                               path.pathRcInfo.rcHitIndex == NO_RCINDEX &&
                               path.pathRcInfo.pathLength == originPath.pathRcInfo.pathLength &&
                               path.pathRcInfo.isLastVertexNEE == originPath.pathRcInfo.isLastVertexNEE;
                jacobian = shiftSucceed ? 1.0f : 0.0f;
                result = shiftSucceed ? float4(path.radiance, 1.0f) : float4(0.0f);
            }
            else
            {
                shiftSucceed = shiftSucceed &&
                               (path.pathRcInfo.rcHitIndex == NO_RCINDEX) &&
                               hit.isValid() &&
                               !path.terminated;
            }

        }

        if (originPath.pathRcInfo.rcHitIndex != NO_RCINDEX && shiftSucceed)
        {
            // ShadingData dstVertexKSd;
            {
                uint INIT_SAMPLE_COUNT = sg.getCount();
                ReconnectionData reconnectionData;
                reconnectionData.dstVertexKMinusOne = hit;
                reconnectionData.dstVertexKMinusOneRay = ray;
                reconnectionData.dstVertexK = shiftedPath.pathRcInfo.rcHit;
                reconnectionData.isDstKExists = reconnectionData.dstVertexK.isValid();
                reconnectionData.isLastVertexNEE = shiftedPath.pathRcInfo.isLastVertexNEE;
                reconnectionData.isVertexKLight = (shiftedPath.pathRcInfo.pathLength == shiftedPath.pathRcInfo.rcHitIndex);
                reconnectionData.bsdfComponentKMinusOne = shiftedPath.pathRcInfo.vertexKMinusOneBsdfComponent;
                reconnectionData.bsdfComponentK = shiftedPath.pathRcInfo.vertexKBsdfComponent;
                reconnectionData.vertexKWi = shiftedPath.pathRcInfo.rcWi;

                float3 tmpThroughput;
                float tmpJacobian;
                float tmpRcJacobian;
                bool canConnect = computeReconnectionThroughputAndJacobian(rtq,
                                                                           shiftedPath.pathRcInfo,
                                                                           reconnectionData,
                                                                           path,
                                                                           tmpThroughput,
                                                                           tmpJacobian,
                                                                           tmpRcJacobian );
                jacobian = tmpJacobian;
                path.throughput *= tmpThroughput;
                // uint sampleConsumed = sg.getCount() - INIT_SAMPLE_COUNT;
                // consumeRng(sg, 2 * kNumOfSamplesPerHit - sampleConsumed);
                // ShadingData dstVertexKMinusOneSd = loadShadingData(reconnectionData.dstVertexKMinusOne,
                //                                             reconnectionData.dstVertexKMinusOneRay,
                //                                             ExplicitLodTextureSampler(0.0f));
                // float3 dstVertexKPosW = dstVertexKMinusOneSd.computeRayOrigin(
                //                                     !isTransmission(shiftedPath.pathRcInfo.vertexKMinusOnePathFlags));
                // dstVertexKSd = loadShadingData(reconnectionData.dstVertexK,
                //                                Ray(dstVertexKPosW, float3(0.0f)),
                //                                ExplicitLodTextureSampler(0.0f));

                shiftSucceed = shiftSucceed && canConnect;
                // bool isVertexKConnectToNEE = shiftedPath.pathRcInfo.rcHitIndex + 1 == shiftedPath.pathRcInfo.pathLength &&
                //                              shiftedPath.pathRcInfo.isLastVertexNEE;
                // bool isPathEnded = shiftedPath.pathRcInfo.pathLength <= shiftedPath.pathRcInfo.rcHitIndex;
                // if (isPathEnded || isVertexKConnectToNEE)
                // {
                path.radiance = path.throughput;
                result.xyz = path.radiance;
                // }
            }
            if (!shiftSucceed)
            {
                print("Step 2 failed:", shiftSucceed);
            }

            // if (shiftSucceed)
            // {
            //     path.pathRcInfo.rcHitIndex = shiftedPath.pathRcInfo.rcHitIndex;

            //     Ray ray = Ray(dstVertexKSd.computeRayOrigin(!isTransmission(shiftedPath.pathRcInfo.vertexKPathFlags)),
            //                     shiftedPath.pathRcInfo.rcWi);
            //     while (!path.terminated)
            //     {
            //         HitInfo hit;
            //         float hitT;
            //         bool isHit = rtq.traceRay(ray, hit, hitT);
            //         Ray scatterRay;
            //         if (hit.isValid())
            //         {
            //             uint INIT_SAMPLE_COUNT = sg.getCount();
            //             handleHitReplay(ray,
            //                       hit,
            //                       emissiveLightSampler,
            //                       shiftedPath,
            //                       rtq,
            //                       sg,
            //                       scatterRay,
            //                       path);
            //             uint sampleConsumed = sg.getCount() - INIT_SAMPLE_COUNT;
            //             consumeRng(sg, kNumOfSamplesPerHit - sampleConsumed);
            //             ray = scatterRay;
            //         }
            //         else
            //         {
            //             path.terminated = true;
            //         }
            //     }
            //     result.xyz = path.radiance;

            //     shiftSucceed = shiftSucceed &&
            //                    path.pathRcInfo.pathLength == originPath.pathRcInfo.pathLength &&
            //                    path.pathRcInfo.isLastVertexNEE == originPath.pathRcInfo.isLastVertexNEE;
            // }

            if (!shiftSucceed)
            {
                result = float4(0.0f);
                jacobian = 0.0f;
                // print("path.length:", path.length);
                // print("originPath.rcIndex:", originPath.pathRcInfo.rcHitIndex);
                // print("originPath.length:", originPath.pathRcInfo.pathLength);
                // print("originPath.isLastVertexNEE:", originPath.pathRcInfo.isLastVertexNEE);
                // print("path.length:", path.pathRcInfo.pathLength);
            }
        }

        return result;
    }

    float4 areaReSTIRShiftMappingLight<RTQuery : IRTQuery>( const PathSampleRecord originPath,
                                                            const float2 fracPixelLoc,
                                                            bool isPrevFrame,
                                                            RTQuery rtq,
                                                            out float jacobian )
    {
        CameraData cameraData = isPrevFrame ? prevCameraData : gScene.camera.data;

        // For now, we should have a same fractional pixel location.
        // shiftedPath.pathRcInfo.fracPixelLoc = fracPixelLoc;

        VBuffer shiftPrimaryHitInfo;
        shiftPrimaryHitInfo.pixel = uint2(fracPixelLoc);
        computePrimaryHit(cameraData,
                          fracPixelLoc + originPath.pathRcInfo.fracPixelLoc,
                          rtq,
                          shiftPrimaryHitInfo);
        float4 L = float4(0.0f);
        if (shiftPrimaryHitInfo.primaryHitInfo.isValid())
        {
            L = hybridShiftMappingLight(originPath, shiftPrimaryHitInfo.pixel,
                                    shiftPrimaryHitInfo, rtq, jacobian);

        }

        return L;
    }

    float4 hybridShiftMappingLight<RTQuery : IRTQuery>(const PathSampleRecord originPath,
                                                       uint2 pixel,
                                                       VBuffer primaryHitInfo,
                                                       RTQuery rtq,
                                                       out float jacobian)
    {
        float4 result = float4(0.0f);

        SampleGenerator sg = SampleGenerator(originPath.initRandomSeed.xy, originPath.initRandomSeed.z);
        Ray ray = primaryHitInfo.cameraRay;
        HitInfo hit = primaryHitInfo.primaryHitInfo;
        PathVertex path = PathVertex();
        bool shiftSucceed = true;
        {
            Ray scatterRay;

            int rpLength = originPath.pathRcInfo.rcHitIndex == NO_RCINDEX ?
                           originPath.pathRcInfo.pathLength + 1 :
                           originPath.pathRcInfo.rcHitIndex - 1;

            for (int i = 0; i < rpLength; ++i)
            {
                uint INIT_SAMPLE_COUNT = sg.getCount();

                if (hit.isValid())
                {
                    handleHitReplay(ray,
                                    hit,
                                    emissiveLightSampler,
                                    originPath,
                                    rtq,
                                    sg, scatterRay, path);
                    ray = scatterRay;
                }
                else
                {
                    path.terminated = true;
                    break;
                }
                uint sampleConsumed = sg.getCount() - INIT_SAMPLE_COUNT;
                consumeRng(sg, kNumOfSamplesPerHit - sampleConsumed);
                HitInfo curHit;
                float hitT;
                bool isHit = rtq.traceRay(ray, curHit, hitT);
                hit = curHit;
            }

            if (originPath.pathRcInfo.rcHitIndex == NO_RCINDEX)
            {
                shiftSucceed = !originPath.pathRcInfo.isDeltaOnlyPath &&
                               any(path.radiance > 0.0f) &&
                               path.pathRcInfo.rcHitIndex == NO_RCINDEX &&
                               path.pathRcInfo.pathLength == originPath.pathRcInfo.pathLength &&
                               path.pathRcInfo.isLastVertexNEE == originPath.pathRcInfo.isLastVertexNEE;
                jacobian = shiftSucceed ? 1.0f : 0.0f;
                result = shiftSucceed ? float4(path.radiance, 1.0f) : float4(0.0f);
            }
            else
            {
                shiftSucceed = shiftSucceed &&
                               (path.pathRcInfo.rcHitIndex == NO_RCINDEX) &&
                               hit.isValid() &&
                               !path.terminated;
            }

        }

        if (originPath.pathRcInfo.rcHitIndex != NO_RCINDEX && shiftSucceed)
        {
            // ShadingData dstVertexKSd;
            {
                uint INIT_SAMPLE_COUNT = sg.getCount();
                ReconnectionData reconnectionData;
                reconnectionData.dstVertexKMinusOne = hit;
                reconnectionData.dstVertexKMinusOneRay = ray;
                reconnectionData.dstVertexK = originPath.pathRcInfo.rcHit;
                reconnectionData.isDstKExists = reconnectionData.dstVertexK.isValid();
                reconnectionData.isLastVertexNEE = originPath.pathRcInfo.isLastVertexNEE;
                reconnectionData.isVertexKLight = (originPath.pathRcInfo.pathLength == originPath.pathRcInfo.rcHitIndex);
                reconnectionData.bsdfComponentKMinusOne = originPath.pathRcInfo.vertexKMinusOneBsdfComponent;
                reconnectionData.bsdfComponentK = originPath.pathRcInfo.vertexKBsdfComponent;
                reconnectionData.vertexKWi = originPath.pathRcInfo.rcWi;

                float3 tmpThroughput;
                float tmpJacobian;
                float tmpRcJacobian;
                bool canConnect = computeReconnectionThroughputAndJacobianLight(rtq,
                                                                                originPath.pathRcInfo,
                                                                                reconnectionData,
                                                                                path,
                                                                                tmpThroughput,
                                                                                tmpJacobian,
                                                                                tmpRcJacobian );
                jacobian = tmpJacobian;
                path.throughput *= tmpThroughput;
                // uint sampleConsumed = sg.getCount() - INIT_SAMPLE_COUNT;
                // consumeRng(sg, 2 * kNumOfSamplesPerHit - sampleConsumed);
                // ShadingData dstVertexKMinusOneSd = loadShadingData(reconnectionData.dstVertexKMinusOne,
                //                                             reconnectionData.dstVertexKMinusOneRay,
                //                                             ExplicitLodTextureSampler(0.0f));
                // float3 dstVertexKPosW = dstVertexKMinusOneSd.computeRayOrigin(
                //                                     !isTransmission(shiftedPath.pathRcInfo.vertexKMinusOnePathFlags));
                // dstVertexKSd = loadShadingData(reconnectionData.dstVertexK,
                //                                Ray(dstVertexKPosW, float3(0.0f)),
                //                                ExplicitLodTextureSampler(0.0f));

                shiftSucceed = shiftSucceed && canConnect;
                // bool isVertexKConnectToNEE = originPath.pathRcInfo.rcHitIndex + 1 == originPath.pathRcInfo.pathLength &&
                //                              originPath.pathRcInfo.isLastVertexNEE;
                // bool isPathEnded = originPath.pathRcInfo.pathLength <= originPath.pathRcInfo.rcHitIndex;
                // if (isPathEnded || isVertexKConnectToNEE)
                // {
                path.radiance = path.throughput;
                result.xyz = path.radiance;
                // }
            }
            if (!shiftSucceed)
            {
                print("Step 2 failed:", shiftSucceed);
            }

            // if (shiftSucceed)
            // {
            //     path.pathRcInfo.rcHitIndex = shiftedPath.pathRcInfo.rcHitIndex;

            //     Ray ray = Ray(dstVertexKSd.computeRayOrigin(!isTransmission(shiftedPath.pathRcInfo.vertexKPathFlags)),
            //                     shiftedPath.pathRcInfo.rcWi);
            //     while (!path.terminated)
            //     {
            //         HitInfo hit;
            //         float hitT;
            //         bool isHit = rtq.traceRay(ray, hit, hitT);
            //         Ray scatterRay;
            //         if (hit.isValid())
            //         {
            //             uint INIT_SAMPLE_COUNT = sg.getCount();
            //             handleHitReplay(ray,
            //                       hit,
            //                       emissiveLightSampler,
            //                       shiftedPath,
            //                       rtq,
            //                       sg,
            //                       scatterRay,
            //                       path);
            //             uint sampleConsumed = sg.getCount() - INIT_SAMPLE_COUNT;
            //             consumeRng(sg, kNumOfSamplesPerHit - sampleConsumed);
            //             ray = scatterRay;
            //         }
            //         else
            //         {
            //             path.terminated = true;
            //         }
            //     }
            //     result.xyz = path.radiance;

            //     shiftSucceed = shiftSucceed &&
            //                    path.pathRcInfo.pathLength == originPath.pathRcInfo.pathLength &&
            //                    path.pathRcInfo.isLastVertexNEE == originPath.pathRcInfo.isLastVertexNEE;
            // }

            if (!shiftSucceed)
            {
                result = float4(0.0f);
                jacobian = 0.0f;
                // print("path.length:", path.length);
                // print("originPath.rcIndex:", originPath.pathRcInfo.rcHitIndex);
                // print("originPath.length:", originPath.pathRcInfo.pathLength);
                // print("originPath.isLastVertexNEE:", originPath.pathRcInfo.isLastVertexNEE);
                // print("path.length:", path.pathRcInfo.pathLength);
            }
        }

        return result;
    }

    bool computeReconnectionThroughputAndJacobianLight<RTQuery : IRTQuery>( RTQuery rtq,
                                                                            const PathRcInfo pathRcInfo,
                                                                            ReconnectionData reconnectionData,
                                                                            inout PathVertex path,
                                                                            out float3 reconnectionThroughput,
                                                                            out float reconnectionJacobian,
                                                                            out float reconnectionRcJacobian )
    {
        SampleGenerator sg = SampleGenerator(uint2(1), 1);

        bool isVertexKLight = reconnectionData.isVertexKLight;
        bool isVertexKNEE   = isVertexKLight && pathRcInfo.isLastVertexNEE;

        bool isVertexKConnectToNEE = pathRcInfo.rcHitIndex + 1 == pathRcInfo.pathLength && pathRcInfo.isLastVertexNEE;

        // TODO
        ShadingData dstVertexKMinusOneSd = loadShadingData(reconnectionData.dstVertexKMinusOne,
                                                        reconnectionData.dstVertexKMinusOneRay,
                                                        ExplicitLodTextureSampler(0.0f));
        bool isTransmissionKMinusOne = isTransmission(pathRcInfo.vertexKMinusOnePathFlags);
        const IMaterialInstance dstVertexKMinusOneMI = gScene.materials.getMaterialInstance(dstVertexKMinusOneSd,
                                                    ExplicitLodTextureSampler(0.0f));
        float3 dstVertexKMinusOnePosW = dstVertexKMinusOneSd.computeRayOrigin(!isTransmissionKMinusOne);

        ShadingData dstVertexKSd = loadShadingData(reconnectionData.dstVertexK,
                                                    Ray(dstVertexKMinusOnePosW, float3(0.0f)),
                                                    ExplicitLodTextureSampler(0.0f));
        const IMaterialInstance dstVertexKMI = gScene.materials.getMaterialInstance(dstVertexKSd, ExplicitLodTextureSampler(0.0f));
        float3 dstVertexKPosW = dstVertexKSd.posW;
        float3 dstVertexKN = dstVertexKSd.getOrientedFaceNormal();
        float3 dirKMinusOneToK = normalize(dstVertexKPosW - dstVertexKMinusOnePosW);
        float distKMinusOneToK = sqr(length(dstVertexKPosW - dstVertexKMinusOnePosW));
        float cosineKMinusOneToK = abs(dot(dirKMinusOneToK, dstVertexKN));
        bool dstKFrontFacing = dstVertexKSd.frontFacing;

        float geometryTerm = reconnectionData.isDstKExists ? cosineKMinusOneToK / distKMinusOneToK : 1.0f;
        float dstKMinusOnePdfBSDF, dstKMinusOnePdfBSDFForMIS;
        float3 bsdfKMinusOne = dstVertexKMinusOneMI.evalBsdfAndPdf( dstVertexKMinusOneSd, dirKMinusOneToK,
                                    sg, isVertexKNEE ? 0xf : 1 << reconnectionData.bsdfComponentKMinusOne,
                                    dstKMinusOnePdfBSDF, dstKMinusOnePdfBSDFForMIS );

        float lightPdf1 = geometryTerm > 0.0f ? pathRcInfo.lightPdf * pathRcInfo.geometryTerm / geometryTerm : 0.0f;
        float dstPdf1 = isVertexKNEE ? lightPdf1 : dstKMinusOnePdfBSDF;

        bool isDstKMinusOneLight = any(dstVertexKMinusOneMI.getProperties(dstVertexKMinusOneSd).emission > 0.0f);
        if (isDelta(pathRcInfo.vertexKMinusOnePathFlags) || isDelta(pathRcInfo.vertexKPathFlags) || isDstKMinusOneLight)
        {
            reconnectionJacobian = 0.0f;
            reconnectionRcJacobian = 0.0f;
            reconnectionThroughput = float3(0.0f);
            return false;
        }

        reconnectionThroughput = pathRcInfo.emission;
        reconnectionRcJacobian = reconnectionData.isDstKExists ? geometryTerm : 1.0f;
        reconnectionThroughput *= (dstPdf1 > 0.0f ? bsdfKMinusOne / dstPdf1 : 0.0f);
        reconnectionRcJacobian *= (reconnectionData.isDstKExists ? dstPdf1 : 1.0f);
        if (reconnectionData.isDstKExists && reconnectionData.isVertexKLight && !dstKFrontFacing)
        {
            reconnectionThroughput = float3(0.0f);
            reconnectionJacobian = 0.0f;
            reconnectionRcJacobian = 0.0f;
            return false;
        }
        float misKMinusOne = kUseNEE && reconnectionData.isVertexKLight ? balanceHeuristic( 1, isVertexKNEE ? lightPdf1 : dstKMinusOnePdfBSDFForMIS,
                                                                                            1, isVertexKNEE ? dstKMinusOnePdfBSDFForMIS : lightPdf1 ) : 1.0f;
        reconnectionThroughput *= misKMinusOne;

        bool needSecondPart = reconnectionData.isDstKExists && !isVertexKLight;
        float dstKPdfBSDF, dstKPdfBSDFForMIS;
        float3 bsdfK = needSecondPart ? dstVertexKMI.evalBsdfAndPdf( dstVertexKSd, reconnectionData.vertexKWi,
            sg, isVertexKConnectToNEE ? 0xf : 1 << reconnectionData.bsdfComponentK,
            dstKPdfBSDF, dstKPdfBSDFForMIS ) : float3(1.0f);
        float lightPdf2 = pathRcInfo.lightPdf;
        float dstPdf2 = isVertexKConnectToNEE ? lightPdf2 : dstKPdfBSDF;
        float misK = (kUseNEE && isVertexKConnectToNEE) ? balanceHeuristic( 1, isVertexKConnectToNEE ? lightPdf2 : dstKPdfBSDFForMIS,
                                                                          1, isVertexKConnectToNEE ? dstKPdfBSDFForMIS : lightPdf2 ) : 1.0f;
        reconnectionThroughput *= misK;
        reconnectionThroughput *= (needSecondPart ? (bsdfK / dstPdf2) : 1.0f);
        reconnectionRcJacobian *= (needSecondPart ? dstPdf2 : 1.0f);
        reconnectionJacobian = pathRcInfo.rcJacobian > 0.0f ? reconnectionRcJacobian / pathRcInfo.rcJacobian : 0.0f;

        path.lastBsdfSamplingpdf = dstKPdfBSDF;
        path.pdf = dstKPdfBSDFForMIS;
        path.lastBSDFComponent = reconnectionData.bsdfComponentK;
        path.lastNormal = dstVertexKSd.getOrientedFaceNormal();
        path.flags = pathRcInfo.vertexKPathFlags;
        path.lastLobeType = pathRcInfo.vertexKBsdfLobes;
        path.sampleUpperHemisphere = pathRcInfo.vertexKSampleUpperHemisphere;

        bool isVisible = isVertexKNEE ? evalSegmentVisibility(false, dstVertexKMinusOnePosW, computeRayOrigin(dstVertexKPosW, dstVertexKN), rtq, false) :
                                        evalSegmentVisibility(false, dstVertexKMinusOnePosW, dstVertexKPosW, rtq, false);
        if (!reconnectionData.isDstKExists)
        {
            isVisible = false;
        }
        bool isValidShift = !isJacobianInvalid(reconnectionJacobian) && !isIntegrandInvalid(reconnectionThroughput) && isVisible;
        // if (!isValidShift)
        // {
        //     print("reconnectionJacobian:", reconnectionJacobian);
        //     print("dstPdf1:", dstPdf1);
        //     print("emission:", pathRcInfo.emission);
        //     print("lightPdf1:", lightPdf1);
        //     print("bsdfKMinusOnePdf:", dstKMinusOnePdfBSDF);
        //     print("bsdfKMinusOnePdfForMIS", dstKMinusOnePdfBSDFForMIS);
        //     print("misMinusOne:", misKMinusOne);
        //     print("needSecondPart:", needSecondPart);
        //     print("dstPdf2:", dstPdf2);
        //     print("bsdfK:", bsdfK);
        //     print("reconnectionThroughput:", reconnectionThroughput);
        //     print("isVisible:", isVisible);
        // }
        reconnectionJacobian = isValidShift ? reconnectionJacobian : 0.0f;
        reconnectionRcJacobian = isValidShift ? reconnectionRcJacobian : 0.0f;
        reconnectionThroughput = isValidShift ? reconnectionThroughput : float3(0.0f);

        path.length = (uint16_t) pathRcInfo.rcHitIndex;
        if (pathRcInfo.pathLength > pathRcInfo.rcHitIndex && !isVertexKConnectToNEE)
        {
            path.length++;
        }

        float distanceFromKMinusOneToK = length(dstVertexKMinusOnePosW - dstVertexKSd.posW);
        bool isDistanceFar = distanceFromKMinusOneToK > params.sceneRadius * params.distanceThreshold;

        uint dstVertexKMinusOneLobeTypes = dstVertexKMinusOneMI.getLobeTypes(dstVertexKMinusOneSd);
        bool isDstVertexKMinusOneRough = false;
        {
            bool hasDiffuse = (dstVertexKMinusOneLobeTypes & (uint)LobeType::Diffuse) > 0u;
            bool highRoughness = dstVertexKMinusOneMI.getProperties(dstVertexKMinusOneSd).roughness > params.roughnessThreshold;
            isDstVertexKMinusOneRough = hasDiffuse || highRoughness;
        }
        bool canConnect = isValidShift && isDistanceFar && isDstVertexKMinusOneRough;
        // if (!canConnect)
        // {
        //     print("isDistanceFar:", isDistanceFar);
        //     print("isKRough:", isDstVertexKMinusOneRough);
        //     print("isKMinusOneRough:", isDstVertexKMinusOneRough);
        // }
        // bool isPathEnded = pathRcInfo.pathLength <= pathRcInfo.rcHitIndex;
        // if (isPathEnded)
        // {
        //     path.pathRcInfo.pathLength = path.length;
        //     path.terminated = true;

        //     if (isVertexKNEE)
        //     {
        //         path.pathRcInfo.isLastVertexNEE = true;
        //     }
        //     else
        //     {
        //         path.pathRcInfo.isLastVertexNEE = false;
        //     }
        // }
        // else if (isVertexKConnectToNEE)
        // {
        //     path.pathRcInfo.pathLength = path.length + 1;
        //     path.pathRcInfo.isLastVertexNEE = true;
        //     path.terminated = true;
        // }

        return canConnect;
    }

    bool evalSegmentVisibility<RTQuery : IRTQuery>(bool isPrevFrame, const float3 sourceVertex,
                                                   const float3 targetVertex,
                                                   RTQuery rtq,
                                                   bool isDirection = false)
    {
        float3 dir = isDirection ? targetVertex : targetVertex - sourceVertex;
        float tMax = isDirection ? kRayTMax : 0.9999f;

        Ray ray = Ray(sourceVertex, dir, 0.0f, tMax); // TODO: use offset position instead

        return rtq.traceVisibilityRay(ray);
    }

}
