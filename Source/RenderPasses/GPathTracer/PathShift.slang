// Copyright @yucwang 2024

import Scene.HitInfo;
import Utils.Sampling.SampleGenerator;
import Utils.Math.Ray;
import Utils.Math.BitTricks;
import Utils.Math.PackedFormats;

import Utils;

const static uint NO_RCINDEX = 0xff;

// Path RIS State for NEE and MIS
struct PathRisState
{
    float wSum;
    float3 integrand;
    float W;
    float M;

    // path information
    bool isLastVertexNEE;
    float lightPdf;

    PathRcInfo pathRcInfo;

    __init()
    {
        this.wSum = 0.0f;
        this.integrand = float3(0.0f);
        this.W = 0.0f;
        this.M = 0.0f;

        this.isLastVertexNEE = false;
        this.pathRcInfo = PathRcInfo();
    }

    [mutating]
    bool add(float3 integrand, float m, inout SampleGenerator sg)
    {
        float pHat = targetFunc(integrand);
        float w = pHat * m;
        if (isnan(w) || isinf(w))
            w = 0.0f;
        this.wSum += w;
        bool forceSelect = (this.M == 0.0f);
        this.M += m;

        bool selected = (this.wSum * sampleNext1D(sg) < w);
        if (selected)
        {
            this.integrand = integrand;
        }

        return selected;
    }

    [mutating]
    bool addBSDFVertex(float3 integrand, float m, uint length, inout SampleGenerator sg)
    {
        bool isSelected = this.add(integrand, m, sg);
        if (isSelected)
        {
            this.isLastVertexNEE = false;
        }

        return isSelected;
    }

    [mutating]
    bool addNEEVertex(float3 integrand, float m, uint length, inout SampleGenerator sg)
    {
        bool isSelected = this.add(integrand, m, sg);
        if (isSelected)
        {
            this.isLastVertexNEE = true;
        }

        return isSelected;
    }

    [mutating]
    void finalize()
    {
        if (this.M == 0.0f)
        {
            this.W = 0.0f;
            return;
        }

        float w = targetFunc(this.integrand);
        this.W = w * this.M > 0.0f ? this.wSum / w : 0.0f;
        if (isnan(this.W) || isinf(this.W))
        {
            this.W = 0.0f;
            return;
        }
    }
}

struct PathRcInfo
{
    // uint2 fracPixelLocData;
    // float2 fracPixelLoc;
    float rcJacobian;
    // uint pathLength;
    float lightPdf;
    float geometryTerm;
    uint rcWiData;
    float3 emission;

    uint data;
    uint2 data1;
    PackedHitInfo packedRcHit;
    // Total 56 bytes

    __init()
    {
        this.fracPixelLoc = float2(0.0f);
        this.rcJacobian = 0.0f;
        this.rcHitIndex = NO_RCINDEX;
        this.pathLength = 0u;
        this.isLastVertexNEE = false;
        this.lightPdf = 0.0f;
        this.geometryTerm = 0.0f;
        this.emission = float3(0.0f);
        this.rcWi = float3(0.0f);
        this.vertexKMinusOneBsdfComponent = 0u;
        this.vertexKBsdfComponent = 0u;
        this.vertexKMinusOnePathFlags = 0u;
        this.vertexKPathFlags = 0u;
        this.vertexKBsdfLobes = 0u;
        this.vertexKSampleUpperHemisphere = true;
        this.isDeltaOnlyPath = true;
        this.packedRcHit = PackedHitInfo(0);
    }

    [mutating]
    inline void update(const PathRcInfo other)
    {
        this.fracPixelLoc = other.fracPixelLoc;
        this.rcJacobian = other.rcJacobian;
        this.pathLength = other.pathLength;
        this.lightPdf = other.lightPdf;
        this.geometryTerm = other.geometryTerm;
        this.emission = other.emission;
        this.rcWi = other.rcWi;
        this.data = other.data;
        this.packedRcHit = other.packedRcHit;
    }

    property uint pathLength
    {
        get
        {
            return (data1.x >> 20) & 0x000000ff;
        }
        set
        {
            this.data1.x = (this.data1.x & 0xf00fffff) | (newValue << 20);
        }
    }

    property float2 fracPixelLoc
    {
        get
        {
            return uint2ToFloat2(uint2(this.data1.x & 0x000fffff, this.data1.y));
        }
        set
        {
            uint2 newVal = float2ToUint2(newValue);
            this.data1.x = (this.data1.x & 0xfff00000) | (newVal.x & 0x000fffff);
            this.data1.y = newVal.y;
        }
    }

    property float3 rcWi
    {
        get
        {
            return decodeNormal2x16(this.rcWiData);
        }
        set
        {
            this.rcWiData = encodeNormal2x16(newValue);
        }
    }

    property uint rcHitIndex
    {
        get
        {
            return ((this.data >> 24) & 0xff);
        }

        set
        {
            this.data = ((this.data & 0x00ffffff) | (newValue << 24));
        }
    }

    property uint vertexKBsdfLobes
    {
        get
        {
            return ((this.data >> 16) & 0xff);
        }

        set
        {
            this.data = ((this.data & 0xff00ffff) | (newValue << 16));
        }
    }

    property uint vertexKPathFlags
    {
        get
        {
            return ((this.data >> 8) & 0xf);
        }

        set
        {
            this.data = ((this.data & 0xfffff0ff) | (newValue << 8));
        }
    }

    property uint vertexKMinusOnePathFlags
    {
        get
        {
            return ((this.data >> 12) & 0xf);
        }

        set
        {
            this.data = ((this.data & 0xffff0fff) | (newValue << 12));
        }
    }

    property uint vertexKBsdfComponent
    {
        get
        {
            return ((this.data >> 6) & 0x3);
        }

        set
        {
            this.data = (this.data & 0xffffff3f) | (newValue << 6);
        }
    }

    property uint vertexKMinusOneBsdfComponent
    {
        get
        {
            return ((this.data >> 4) & 0x3);
        }

        set
        {
            this.data = (this.data & 0xffffffcf) | (newValue << 4);
        }
    }

    property bool isDeltaOnlyPath
    {
        get
        {
            return (this.data & 0x1) > 0;
        }

        set
        {
            this.data = (this.data & 0xfffffffe) | (newValue ? 0x1 : 0x0);
        }
    }

    property bool vertexKSampleUpperHemisphere
    {
        get
        {
            return ((this.data >> 1) & 0x1) > 0;
        }

        set
        {
            this.data = (this.data & 0xfffffffd) | (newValue ? 0x2 : 0x0);
        }
    }

    property bool isLastVertexNEE
    {
        get
        {
            return ((this.data >> 2) & 0x1) > 0;
        }

        set
        {
            this.data = (this.data & 0xfffffffb) | (newValue ? 0x4 : 0x0);
        }
    }

    property HitInfo rcHit
    {
        get
        {
            return HitInfo(packedRcHit);
        }

        set
        {
            this.packedRcHit = newValue.data;
        }
    }
}

// Path record for shift mapping
struct PathSampleRecord
{
    // The random seed for generating this path
    uint2 initRandomSeedData;

    // For multi-point methods
    float innerJacobian;

    uint data;

    PathRcInfo pathRcInfo;
    // Total 36 + 64 bytes

    __init(uint2 pixelID, uint randomSeed)
    {
        this.initRandomSeed = uint3(pixelID, randomSeed);
        this.index = 0;
        this.pathRcInfo      = PathRcInfo();
        this.innerJacobian = 0.0f;
        this.data = 0u;
    }

    [mutating]
    inline void update(const PathSampleRecord other)
    {
        this.initRandomSeed = other.initRandomSeed;
        this.pathRcInfo.update(other.pathRcInfo);
        this.innerJacobian = other.innerJacobian;
        this.data = other.data;
    }

    property uint3 initRandomSeed
    {
        get
        {
            static const uint mask = 0x0000ffff;
            return uint3((this.initRandomSeedData.x >> 16) & mask,
                         (this.initRandomSeedData.x & mask),
                         this.initRandomSeedData.y);
        }
        set
        {
            static const uint mask = 0x0000ffff;
            this.initRandomSeedData.x = (newValue.x << 16) | (newValue.y & mask);
            this.initRandomSeedData.y = newValue.z;
        }
    }

    property uint index
    {
        get { return this.data & 0x1; }
        set
        {
            this.data = (this.data & 0xfffffffe) | (newValue & 0x1);
        }
    }

    property bool isLastVertexNEE
    {
        get { return ((this.data >> 1) & 0x1) > 0 ? true : false; }
        set
        {
            this.data = (this.data & 0xfffffffd) | (newValue ? 0x2 : 0x0);
        }
    }

    property uint pathLength
    {
        get { return (data >> 4) & 0xf; }
        set
        {
            this.data = (this.data & 0xffffff0f) | (newValue << 4);
        }
    }

    property float2 sampleBaseFracPixelLoc
    {
        get { return this.pathRcInfo.fracPixelLoc; }
        set
        {
            this.pathRcInfo.fracPixelLoc = newValue;
        }
    }

    // property float2 sampleShiftFracPixelLoc
    // {
    //     get { return this.index == 0 ? this.shiftPathRcInfo.fracPixelLoc : this.pathRcInfo.fracPixelLoc; }
    //     set
    //     {
    //         if (this.index == 0)
    //             this.shiftPathRcInfo.fracPixelLoc = newValue;
    //         else if (this.index == 1)
    //             this.pathRcInfo.fracPixelLoc = newValue;
    //     }
    // }
}

struct LastVertexState
{
    uint data;

    __init(uint _data)
    {
        data = _data;
    }

    [mutating]
    void Init(bool isCurrentVertexFarFromPrev, uint lastBSDFComponent,
              bool isLastVertexDelta,
              bool isLastVertexTransmission,
              bool isLastVertexRough,
              bool isVertexClassifiedAsRoughForNEE)
    {
        data = 0;
        data |= (uint)isLastVertexDelta;
        data |= (uint)isLastVertexTransmission << 1;
        data |= (uint)lastBSDFComponent << 2;
        data |= (uint)isLastVertexRough << 4;
        data |= (uint)isCurrentVertexFarFromPrev << 5;
        data |= (uint)isVertexClassifiedAsRoughForNEE << 6;
    }

    property bool isLastVertexDelta
    {
        get
        {
            return data & 1;
        }
        set
        {
            data &= ~1;
            data |= newValue;
        }
    }

    property bool isLastVertexTransmission
    {
        get
        {
            return (data >> 1) & 1;
        }
        set
        {
            data &= ~(1 << 1);
            data |= newValue << 1;
        }
    }

    property uint lastBSDFComponent
    {
        get
        {
            return (data >> 2) & 3u;
        }
        set
        {
            data &= ~(3 << 2);
            data |= newValue << 2;
        }
    }

    property bool isLastVertexRough
    {
        get
        {
            return (data >> 4) & 1;
        }
        set
        {
            data &= ~(1 << 4);
            data |= newValue << 4;
        }
    }

    property bool isCurrentVertexFarFromPrev
    {
        get
        {
            return (data >> 5) & 1;
        }
        set
        {
            data &= ~(1 << 5);
            data |= newValue << 5;
        }
    }

    property bool isVertexClassifiedAsRoughForNEE
    {
        get
        {
            return (data >> 6) & 1;
        }
        set
        {
            data &= ~(1 << 6);
            data |= newValue << 6;
        }
    }
}

struct ReconnectionData
{
    uint bsdfComponentKMinusOne;
    uint bsdfComponentK;
    HitInfo dstVertexKMinusOne;
    Ray dstVertexKMinusOneRay;
    HitInfo dstVertexK;
    float3 vertexKWi;
    bool isVertexKLight;
    bool isLastVertexNEE;
    bool isDstKExists;
}

void randomReplayGPathShift(const PathSampleRecord originPath,
                            out PathSampleRecord shiftedPath,
                            out float jacobian)
{
    shiftedPath.update(originPath);
    jacobian = 1.0f;
}

// Shift a path by random replay.
void randomReplayShift(const PathSampleRecord originPath,
                       out PathSampleRecord shiftedPath,
                       out float jacobian)
{
    shiftedPath.update(originPath);
    jacobian = 1.0f;
}
