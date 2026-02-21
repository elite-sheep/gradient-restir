// Copyright @yucwang 2024

#include "Utils/Math/MathConstants.slangh"

import Scene.HitInfo;
import Scene.ShadingData;
import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGenerator;

import PathShift;
import PathState;

// Reservoir for path sample reuse with RIS
struct PathReservoir
{
    // Sample confidence
    float C;
    // The sum of weight of the samples
    float wSum;
    // The function value
    float3 integrand;
    // The weight of current selected sample
    float w;
    uint pathID;
    // 24 + 108 bytes

    __init()
    {
        this.C = 0.0f;
        this.wSum = 0.0f;
        this.integrand = float3(0.0f);
        this.w = 0.0f;
        this.pathID = 0;
    }

    [mutating]
    void init()
    {
        this.C = 0.0f;
        this.wSum = 0.0f;
        this.integrand = float3(0.0f);
        this.w = 0.0f;
        this.pathID = 0;
    }

    [mutating]
    void init(const PathReservoir other)
    {
        this.C = other.C;
        this.wSum = other.wSum;
        this.integrand = other.integrand;
        this.w = other.w;
        this.pathID = other.pathID;
    }

    [mutating]
    bool addCandidateSample(const uint pathID,
                            float3 integrand,
                            float pHatI,
                            float mi,
                            float Wi,
                            float u)
    {
        // if (isnan(pHatI) || isinf(pHatI))
        //     return false;
        float w = mi * pHatI * Wi;
        if (isnan(w) || isinf(w) || w < 0.0f || any(isnan(integrand)))
            return false;
        this.C += 1.0f;
        this.wSum += w;

        bool sampleSelected = (u * this.wSum < w) || this.C == 1.0f;
        this.pathID = sampleSelected ? pathID : this.pathID;
        this.w = sampleSelected ? (isnan(pHatI) || isinf(pHatI)) ? 0.0f : pHatI : this.w;
        this.integrand = sampleSelected ? integrand : this.integrand;

        return sampleSelected;
    }

    [mutating]
    bool addTemporalSample(const uint pathID,
                           float3 integrand,
                           float pHatI,
                           float Ji,
                           float mi,
                           float ci,
                           float Wi,
                           float u)
    {
        // if (isnan(pHatI) || isinf(pHatI))
        //     return false;
        float w = mi * pHatI * Wi * Ji;
        if (isnan(w) || isinf(w) || w < 0.0f || any(isnan(integrand)))
            return false;
        this.wSum = this.wSum + w;
        this.C = this.C + ci;

        bool sampleSelected = (u * this.wSum < w) || this.C == ci;
        this.pathID = sampleSelected ? pathID : this.pathID;
        this.w = sampleSelected ? (isnan(pHatI) || isinf(pHatI)) ? 0.0f : pHatI : this.w;
        this.integrand = sampleSelected ? integrand : this.integrand;

        return sampleSelected;
    }

    [mutating]
    bool addSpatialSample(const uint pathID,
                           float3 integrand,
                           float pHatI,
                           float JI,
                           float mi,
                           float ci,
                           float Wi,
                           float u)
    {
        // if (isnan(pHatI) || isinf(pHatI))
        //     return false;
        float w = mi * pHatI * Wi * JI;
        if (isnan(w) || isinf(w) || w < 0.0f || any(isnan(integrand)))
            return false;
        this.wSum = this.wSum + w;
        this.C = this.C + ci;

        bool sampleSelected = (u * this.wSum < w);
        this.pathID = sampleSelected ? pathID : this.pathID;
        this.w = sampleSelected ? (isnan(pHatI) || isinf(pHatI)) ? 0.0f : pHatI : this.w;
        this.integrand = sampleSelected ? integrand : this.integrand;

        return sampleSelected;
    }

    [mutating]
    void finalizeInitialCandidates()
    {
    }

    [mutating]
    void finalizeTemporalReuse(float cCap)
    {
        this.C = min(this.C, cCap);
    }

    [mutating]
    void finalizeSpatialReuse(float cCap)
    {
        this.C = min(this.C, cCap);
    }

    [mutating]
    void reset()
    {
        this.C = 0.0f;
        this.wSum = 0.0f;
        this.integrand = float3(0.0f);
        this.w = 0.0f;
    }

    property bool isNullSample
    {
        get
        {
            return isnan(this.W) || isinf(this.W) || this.C <= 0.0f;
        }
    }

    property float W
    {
        get
        {
            return (!isnan(this.w) && !isinf(this.w) && this.w > 0) ? this.wSum / this.w : 0.0f;
        }
    }
}
