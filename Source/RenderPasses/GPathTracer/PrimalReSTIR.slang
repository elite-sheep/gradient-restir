// Copyrigth @yucwang 2024

import Scene.HitInfo;
import Scene.RaytracingInline;
import Scene.ShadingData;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

import GPathTracer;
import PathShift;
import PathState;
import Params;
import RISDataTypes;
import StaticParams;
import Utils;

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

struct PrimalReSTIR
{
    ParameterBlock<GPathTracer> gPathTracer;
    RWStructuredBuffer<PathReservoir> primalReservoirs;
    StructuredBuffer<PathSampleRecord> basePaths;

    Texture1D<float2> neighborOffsets;
    Texture2D<float3> primaryEmission;

    bool shouldSpatialReuse(int iNeighbour, uint neighbourValidMask)
    {
        return iNeighbour == -1 ? true : getBit(neighbourValidMask, iNeighbour);
    }

    int2 getNextNeighborPixel(const uint startIndex, uint2 pixel, int i)
    {
        float gGatherRadius = gPathTracer.params.spatialGatherRadius;
        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * gGatherRadius);

        return neighborPixel;
    }

    float4 evaluate(uint2 pixel)
    {
        uint bufferIndex = getBufferOffset(pixel, gPathTracer.params.frameDim, 2);

        float3 finalColor = primaryEmission[pixel] +
                            primalReservoirs[bufferIndex].integrand * primalReservoirs[bufferIndex].W;

        return float4(finalColor, 1.0f);
    }
}
