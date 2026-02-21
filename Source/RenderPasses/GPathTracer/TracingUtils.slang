/***************************************************************************
 # Copyright (c) 2023-24, Yu-Chen Wang. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
import Rendering.Materials.IMaterialInstance;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Scene.Raytracing;
import Scene.RaytracingInline;
import Scene.Intersection;
import Scene.ShadingData;
import Utils.Geometry.GeometryHelpers;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;

import StaticParams;
import Params;
import PathState;
import Utils;

#include "Utils/Math/MathConstants.slangh"

// Light sample
struct LightSample
{
    // Lighting data
    float3 lightPos;
    float3 lightNormal;
    float3 Le;
    float3 Li;
    float pdf;
    float geometryTerm;

    // For visibility test
    float3 visOrigin;
    float3 visDir;
    float visDistance;

    Ray getVisibilityRay()
    {
        return Ray(visOrigin, visDir, 0.f, visDistance);
    }
}

float balanceHeuristic(float m1, float w1, float m2, float w2)
{
    float q0 = m1 * w1;
    float q1 = m2 * w2;
    return (q0 + q1 > 0) ? q0 / (q0 + q1) : 0.0f;
}

ShadingData loadShadingData(const HitInfo hit, const Ray ray, const ITextureSampler lod)
{
    VertexData v = {};
    uint materialID = {};

    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -ray.dir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
    }
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, ray.origin, ray.dir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
    }

    float3 viewDir = all(ray.dir == 0.0f) ? normalize(ray.origin - v.posW) : -ray.dir;
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, viewDir, lod);

    return sd;
}

bool sampleEmissive(ShadingData sd,
                    EmissiveLightSampler emissiveSampler,
                    bool upperHemisphere,
                    inout SampleGenerator sg,
                    out LightSample ls,
                    out HitInfo neeHitInfo )
{
    ls = {};

    TriangleLightSample tls;
    if (!emissiveSampler.sampleLight(sd.posW, sd.getOrientedFaceNormal(),
                                     upperHemisphere, sg, tls, false))
    {
        neeHitInfo = HitInfo();
        return false;
    }

    // return sample
    ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
    ls.Le = tls.pdf > 0.f ? tls.Le : float(0);
    ls.pdf = tls.pdf;
    ls.lightPos = tls.posW;
    ls.lightNormal = tls.normalW;
    ls.visOrigin = computeRayOrigin(sd.posW, sd.getOrientedFaceNormal());
    float3 visLight = computeRayOrigin(tls.posW, tls.normalW);
    float3 toLight = visLight - ls.visOrigin;
    ls.visDir = normalize(toLight);
    ls.visDistance = length(toLight);

    float3 wo = normalize(ls.lightPos - ls.visOrigin);
    float dis = length(ls.lightPos - ls.visOrigin);
    ls.geometryTerm = abs(dot(ls.lightNormal, -wo)) / sqr(dis);
    // ls.geometryTerm = tls.geometryTerm;

    TriangleHit neeTriangleHit;
    uint2 meshPrimID = gScene.lightCollection.getSceneMeshPrimID(tls.triangleIndex);
    GeometryInstanceID neeGeoInstanceId;
    neeGeoInstanceId.index = meshPrimID.x;
    neeTriangleHit.instanceID = neeGeoInstanceId;
    neeTriangleHit.primitiveIndex = meshPrimID.y;
    neeTriangleHit.barycentrics = tls.uv;
    neeHitInfo = HitInfo(neeTriangleHit);

    return any(ls.Li > 0.0f);
}

// Handle hit when replaying the path
