// Copyright @yucwang 2024

import Rendering.Materials.LobeType;
import Scene.HitInfo;
import Scene.Raytracing;
import Scene.ShadingData;
import Utils.Sampling.SampleGenerator;
import PathShift;

enum class PathFlags
{
    transmission = 0x0001, ///< Scatter ray went through a transmission event.
    specular = 0x0002,     ///< Scatter ray went through a specular event.
    delta = 0x0004,        ///< Scatter ray went through a delta event.
    rough = 0x0008,        ///< classified as a rough path vertex for ReSTIR PT
};

struct VBuffer
{
    uint2 pixel;
    Ray cameraRay;
    PackedHitInfo packedPrimaryHitInfo;
    float3 albedo;

    __init()
    {
        this.pixel = uint2(0, 0);
        this.cameraRay = Ray(0.0f, 0.0f, 0.0f, 0.0f);
        this.albedo = float3(0.0f);
    }

    property HitInfo primaryHitInfo
    {
        get
        {
            return HitInfo(this.packedPrimaryHitInfo);
        }

        set
        {
            this.packedPrimaryHitInfo = newValue.data;
        }
    }
};

// Record the state of path
struct PathVertex
{
    // Select a path from the path tree
    PathRisState risState;
    float3 radiance;
    float3 throughput;
    uint16_t length;
    bool terminated;

    uint flags;
    uint lastFlags;

    // For MIS
    float3 lastNormal;
    float lastBsdfSamplingpdf;
    uint lastLobeType;
    uint lastBSDFComponent;
    float pdf;

    bool sampleUpperHemisphere;

    // For hybrid shift mapping
    LastVertexState lastVertexState;
    PathRcInfo pathRcInfo;

    __init()
    {
        this.risState = PathRisState();
        this.radiance = float3(0.f);
        this.throughput = float3(1.f, 1.f, 1.f);
        this.length = 0;
        this.terminated = false;
        this.lastBsdfSamplingpdf = 0.f;
        this.lastNormal = float3(0.0f);
        this.lastLobeType = (uint)LobeType::None;
        this.lastBSDFComponent = 0u;
        this.lastVertexState = LastVertexState(0u);
        this.pathRcInfo = PathRcInfo();
        this.flags = 0u;
        this.lastFlags = 0u;
        this.sampleUpperHemisphere = true;
    }

    [mutating]
    bool addBSDFVertex(float3 integrand, float m,
                       inout SampleGenerator sg)
    {
        return this.risState.addBSDFVertex(integrand, m, this.length, sg);
    }

    [mutating]
    bool addNEEVertex(float3 integrand, float m,
                      inout SampleGenerator sg)
    {
        return this.risState.addNEEVertex(integrand, m, this.length, sg);
    }

    bool hasFlag(PathFlags flag)
    {
        return (this.flags & (uint)flag) != 0;
    }

    [mutating]
    void setFlag(PathFlags flag, bool value = true)
    {
        this.flags = (value) ? (this.flags | (uint)flag) : (this.flags & ~(uint)flag);
    }

    [mutating]
    void clearFlags()
    {
        this.flags = 0u;
    }

    bool isTransmission() { return hasFlag(PathFlags.transmission); }
    bool isSpecular() { return hasFlag(PathFlags.specular); }
    bool isDelta() { return hasFlag(PathFlags.delta); }
    bool isRough() { return hasFlag(PathFlags::rough); }

    [mutating] void setTransmission(bool value = true) { setFlag(PathFlags::transmission, value); };
    [mutating] void setSpecular(bool value = true) { setFlag(PathFlags::specular, value); };
    [mutating] void setDelta(bool value = true) { setFlag(PathFlags::delta, value); };
    [mutating] void setRough(bool value = true) { setFlag(PathFlags::rough, value); };
};

bool isTransmission(uint flags)
{
    return hasFlag(flags, PathFlags.transmission);
}

bool isSpecular(uint flags)
{
    return hasFlag(flags, PathFlags.specular);
}

bool isDelta(uint flags)
{
    return hasFlag(flags, PathFlags.delta);
}

bool isRough(uint flags)
{
    return hasFlag(flags, PathFlags::rough);
}

bool hasFlag(uint flags, PathFlags flag)
{
    return (flags & (uint)flag) != 0;
}

void setFlag(uint flags, PathFlags flag, bool value = true)
{
    flags = (value) ? (flags | (uint)flag) : (flags & ~(uint)flag);
}
