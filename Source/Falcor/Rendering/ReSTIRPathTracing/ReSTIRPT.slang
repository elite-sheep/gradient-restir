/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#pragma once
#include "RetraceScheduleDefinition.slangh"
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

#ifndef HOST_CODE
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Math.PackedFormats;
//import RenderPasses.InternalPathTracer.PathState;
import Scene.Material.MaterialData;
import Scene.Material.MaterialTypes;
import PathReservoir;
import StaticParams;
//import Params;

#ifndef TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE
#define TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE 0
#endif

#ifndef USE_RESERVOIR_COMPRESSION
#define USE_RESERVOIR_COMPRESSION 0
#endif

#ifndef RETRACE_SCHEDULE_TYPE
#define RETRACE_SCHEDULE_TYPE RETRACE_SCHEDULE_NAIVE
#endif

#ifndef SEPARATE_NEE_RESAMPLING
#define SEPARATE_NEE_RESAMPLING 0
#endif

#ifndef USE_PREV_FRAME_SCENE_DATA
#define USE_PREV_FRAME_SCENE_DATA 0
#endif

#endif

/** ReSTIR PT.

    This exposes the data structures needed to query ReSTIR PT Reservoirs.
*/
struct ReSTIRPT
{
    enum class LocalStrategy
    {
        None = 0x0,
        RoughnessCondition = 0x1,
        DistanceCondition = 0x2,
    };

    enum class RetraceScheduleType
    {
        Naive = RETRACE_SCHEDULE_NAIVE,
        Compact = RETRACE_SCHEDULE_COMPACT
    };

    enum class ShiftMapping
    {
        Reconnection = 0,
        RandomReplay = 1,
        Hybrid = 2,
    };

    enum class SubpathMIS
    {
        Binary = 0,
        P = 1,
        PHat = 2
    };

    enum class KNNAdaptiveRadiusType
    {
        NonAdaptive = 0,
        RayCone = 1,
    };

    struct ShiftMappingSettings
    {
        uint localStrategyType = 0x3;
        float jacobianRejectionThreshold = 10.f;
        float specularRoughnessThreshold = 0.2f;
        float nearFieldDistanceThreshold = 0.03f; // TODO: replace with path footprint
        bool rejectShiftBasedOnJacobian = false;
        #ifdef HOST_CODE
        template<typename Archive>
        void serialize(Archive& ar)
        {
            ar("localStrategyType", localStrategyType);
            ar("jacobianRejectionThreshold", jacobianRejectionThreshold);
            ar("specularRoughnessThreshold", specularRoughnessThreshold);
            ar("nearFieldDistanceThreshold", nearFieldDistanceThreshold);
            ar("rejectShiftBasedOnJacobian", rejectShiftBasedOnJacobian);
        }
        #endif
    };

    struct SubpathReuseSettings
    {
        bool directlyOutputColor = true;
        // for spatial subpath reuse, doesn't shift the full path first
        int numIntegrationPrefixes = 4;
        bool generateCanonicalSuffixForEachPrefix = false;
        int subpathMISKind = 2;
        int suffixSpatialNeighborCount = 3;
        float suffixSpatialReuseRadius = 30.f;
        int suffixSpatialReuseRounds = 1; // high setting: 3
        bool suffixTemporalReuse = true;
        int temporalHistoryLength = 50;
        int prefixNeighborSearchRadius = 30; 
        int prefixNeighborSearchNeighborCount = 50; 
        int finalGatherSuffixCount = 1;
        bool useTalbotMISForGather = true;
        float nonCanonicalWeightMultiplier = 1;
        bool disableCanonical = false;
        bool adaptivePrefixLength = true;
        bool avoidSpecularPrefixEndVertex = true;
        bool avoidShortPrefixEndSegment = true;
        float shortSegmentThreshold = 0.01f;
        bool compressNeighborSearchKey = true;
        float knnSearchRadiusMultiplier = 2.f;
        uint knnSearchAdaptiveRadiusType = 1; // 0 -- not adaptive, 1 -- ray cone
        bool knnIncludeDirectionSearch = false;
        bool useMMIS = false;
        #ifdef HOST_CODE
        template<typename Archive>
        void serialize(Archive& ar)
        {
            ar("directlyOutputColor", directlyOutputColor);
            ar("numIntegrationPrefixes", numIntegrationPrefixes);
            ar("generateCanonicalSuffixForEachPrefix", generateCanonicalSuffixForEachPrefix);
            ar("subpathMISKind", subpathMISKind);
            ar("suffixSpatialNeighborCount", suffixSpatialNeighborCount);
            ar("suffixSpatialReuseRadius", suffixSpatialReuseRadius);
            ar("suffixSpatialReuseRounds", suffixSpatialReuseRounds);
            ar("suffixTemporalReuse", suffixTemporalReuse);
            ar("temporalHistoryLength", temporalHistoryLength);
            ar("prefixNeighborSearchRadius", prefixNeighborSearchRadius);
            ar("prefixNeighborSearchNeighborCount", prefixNeighborSearchNeighborCount);
            ar("finalGatherSuffixCount", finalGatherSuffixCount);
            ar("useTalbotMISForGather", useTalbotMISForGather);
            ar("nonCanonicalWeightMultiplier", nonCanonicalWeightMultiplier);
            ar("disableCanonical", disableCanonical);
            ar("adaptivePrefixLength", adaptivePrefixLength);
            ar("avoidSpecularPrefixEndVertex", avoidSpecularPrefixEndVertex);
            ar("avoidShortPrefixEndSegment", avoidShortPrefixEndSegment);
            ar("shortSegmentThreshold", shortSegmentThreshold);
            ar("compressNeighborSearchKey", compressNeighborSearchKey);
            ar("knnSearchRadiusMultiplier", knnSearchRadiusMultiplier);
            ar("knnSearchAdaptiveRadiusType", knnSearchAdaptiveRadiusType);
            ar("knnIncludeDirectionSearch", knnIncludeDirectionSearch);
            ar("useMMIS", useMMIS);
        }
        #endif
    };

    ShiftMappingSettings settings;
    SubpathReuseSettings subpathSettings;

    uint numSpatialRounds = 1;
    bool additionalSpatial = false;
    uint numAdditionalSpatialRounds = 1;
    uint suffixSpatialRounds = 1;
    int vertexBufferNEEOffset = 0;
    int minimumPrefixLength = 0;
    int samplesPerPixel = 1;
    bool subpathReuse = true;
    uint shiftMapping = 1;

#ifndef HOST_CODE

    uint sgCount()
    {
        uint maxSpp = samplesPerPixel > 0 ? samplesPerPixel : 128;
        // for each additional integration prefix, we need 1(Tracing) + 1 (search) + 1 (suffix streaming) = 3 seeds
        maxSpp += int(subpathReuse) * (3  + suffixSpatialRounds + 1);
        maxSpp += int(subpathReuse) * (subpathSettings.numIntegrationPrefixes - 1) * (1 + 2);
        maxSpp += numSpatialRounds + 1 + int(additionalSpatial) * numAdditionalSpatialRounds;
        return maxSpp;
    }

    uint spatialSgOffset(int spatialRoundId, bool isAdditionalSpatial)
    {
        return samplesPerPixel +
               int(subpathReuse) * (3  + suffixSpatialRounds + 1)
               + int(subpathReuse) * (subpathSettings.numIntegrationPrefixes - 1) * (1 + 2) 
               + (isAdditionalSpatial ? spatialRoundId : int(additionalSpatial) * numAdditionalSpatialRounds + 1 + spatialRoundId);
    }

    uint temporalSgOffset()
    {
        return samplesPerPixel +
               int(subpathReuse) * (3  + suffixSpatialRounds + 1) 
               + int(subpathReuse) * (subpathSettings.numIntegrationPrefixes - 1) * (1 + 2)
               + int(additionalSpatial) * numAdditionalSpatialRounds;
    }

    uint spatialOrTemporalSgOffset(bool isSpatialPass, bool isAdditionalSpatial, int spatialRoundId)
    {
        return samplesPerPixel 
                                                           + int(subpathReuse) * (3  + suffixSpatialRounds + 1)
                                                            + int(subpathReuse) * (subpathSettings.numIntegrationPrefixes - 1) * (1 + 2)
                                                            + (isAdditionalSpatial ? spatialRoundId : (additionalSpatial ? numAdditionalSpatialRounds : 0))
                                                           + (isSpatialPass && !isAdditionalSpatial ? 1 + spatialRoundId : 0);
    }

    uint prefixResamplingSgOffset()
    {
        return samplesPerPixel;
    }

    uint suffixGenerationSgOffset()
    {
        return samplesPerPixel + 1;
    }

    uint addNEEForSuffixSgOffset()
    {
        return samplesPerPixel + 2;
    }

    uint suffixResamplingSgOffset()
    {
        return samplesPerPixel + 3 ;
    }

    uint prefixNeighborSearchOffset(int integrationPrefixId)
    {
        return samplesPerPixel + 3 
         + suffixSpatialRounds
        + integrationPrefixId == 0 ? 0 : 1 + 3 * (integrationPrefixId - 1) + 1;
    }

    uint traceNewPrefixOffset(int integrationPrefixId)
    {
        return samplesPerPixel + 3 
         + suffixSpatialRounds + 1 
        + 3 * (integrationPrefixId - 1);
    }

    uint finalGatherAdditionalOffset(int integrationPrefixId)
    {
        return integrationPrefixId == 0 ? suffixSpatialRounds - 1 : suffixSpatialRounds + 1 + 3 * (integrationPrefixId - 1) + 2;
    }

    uint addNEESgOffset(int curIter)
    {
        return 2 * curIter + 1;
    }

    // static params goes here
    //static const uint kShiftMapping = RESTIRPT_SHIFT_MAPPING;
    static const bool kUseReservoirCompression = USE_RESERVOIR_COMPRESSION;
    static const bool kTemporalUpdateForDynamicScene = TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE;
    static const uint kRetraceScheduleType = RETRACE_SCHEDULE_TYPE;
    static const uint kMaximumRcLength = 15;
    //static const bool kSubpathReuse = SUBPATH_REUSE;
    static const bool kUsePrevFrameSceneData = USE_PREV_FRAME_SCENE_DATA;
    
    RWStructuredBuffer<PathReservoir> pathReservoirs;
    RWStructuredBuffer<PrefixPathReservoir> prefixPathReservoirs;
    RWStructuredBuffer<PrefixGBuffer> prefixGBuffer;

    RWStructuredBuffer<PrefixReservoir> prefixReservoirs;
    RWStructuredBuffer<PrefixInfo> prefixThroughputs;

    RWStructuredBuffer<ReSTIRPTVertexFull> scratchVertices;
    bool needResetTemporalHistory;
    float sceneRadius;
#endif
};


END_NAMESPACE_FALCOR
