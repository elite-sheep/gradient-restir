
/***************************************************************************
# Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
**************************************************************************/
#include "Utils/Math/MathConstants.slangh"
import Scene.RaytracingInline;
// import InternalPathTracer;
import ReSTIRPT;
// import Shift;
import Scene.HitInfo;
import PathReservoir;
import PathTracer;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Color.ColorHelpers;
import LoadShadingData;
import Params;
import SubpathReuse;
import Rendering.Lights.EmissiveLightSamplerInterface;
import Shift;
import ResamplingCommon;

/** Simple scheduler (fullscreen compute pass) using TraceRayInline.
 */
struct SuffixResampling
{
    ReSTIRPathTracerParams params; ///< Runtime parameters.
    // Resources
    ReSTIRPT restirpt; ///< Settings for ReSTIR PT. Only valid if kReSTIRPT == true.

    Texture2D<PackedHitInfo> vbuffer;         ///< Fullscreen V-buffer for the primary hits.
    Texture2D<PackedHitInfo> temporalVbuffer; ///< Fullscreen V-buffer for the primary hits.

    Texture2D<float2> motionVectors;    
    RWStructuredBuffer<PathReservoir> reservoirs;
    StructuredBuffer<PathReservoir> prevReservoirs;

    StructuredBuffer<PrefixGBuffer> prefixGBuffer;
    StructuredBuffer<PrefixGBuffer> prevPrefixGBuffer;

    Texture1D<float2> neighborOffsets;

    StructuredBuffer<ReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<uint> rcBufferOffsets;
    RWStructuredBuffer<NeighborValidMask> neighborValidMask;
    RWTexture2D<float4> outColor;
    StructuredBuffer<uint> temporalNeighborPixels;

    int suffixReuseRoundId;
    int integrationPrefixId;
    bool hasCanonicalSuffix;

    StructuredBuffer<PrefixReservoir> prefixReservoirs;
    
    float3 prevCameraU;
    float3 prevCameraV;
    float3 prevCameraW; ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float prevJitterX;  ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
    float prevJitterY;  ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).

    int curPrefixLength;

    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int i)
    {
        if (i == -1) return pixel;

        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * restirpt.subpathSettings.suffixSpatialReuseRadius);

        return neighborPixel;
    }

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int radius, int i)
    {
        if (i == -1) return pixel;

        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * radius);

        return neighborPixel;
    }

    ReconnectionData fillReconnectionPartialShiftRcDataPrefix(int offset)
    {
        PrefixGBuffer pg = prefixGBuffer[offset];
        return ReconnectionData(pg.hit, pg.wo, float3(1.f));
    }

    ReconnectionData fillReconnectionPartialShiftRcDataPrefixPrevFrame(int offset)
    {
        PrefixGBuffer pg = prevPrefixGBuffer[offset];
        return ReconnectionData(pg.hit, pg.wo, float3(1.f));
    }

    bool isPixelPrefixLastSdValid(int offset)
    {
        PrefixGBuffer pg = prefixGBuffer[offset];
        return pg.hit.isValid();
    }

    bool isPixelNewPrefixLastSdValid(int offset)
    {
        PrefixGBuffer pg = restirpt.prefixGBuffer[offset];
        return pg.hit.isValid();
    }

    void adjustNeighborWeight(inout float M, int numNeighbors)
    {
        if (restirpt.subpathSettings.useMMIS) return;
        M = min(restirpt.subpathSettings.temporalHistoryLength, M);
        M *= restirpt.subpathSettings.nonCanonicalWeightMultiplier;
    }

    void runGatherSuffixes(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
    {
        int2 pixel = dispatchThreadId.xy;
        const int offset = params.getReservoirOffset(pixel);

        SampleGenerator sg = SampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.suffixResamplingSgOffset() 
                                                            + restirpt.finalGatherAdditionalOffset(integrationPrefixId));

        printSetPixel(pixel);
        logSetPixel(pixel);

        bool insideScreen = isValidScreenRegion(params, pixel);
        if (!insideScreen) return;
        bool primaryHitValid = isPrimaryHitValid(vbuffer, pixel, offset);
        if (!primaryHitValid)
        {
            return;
        }

        bool centralPrefixValid = isPixelNewPrefixLastSdValid(offset);

        ShadingData primarySd = getPixelShadingData(params, vbuffer, pixel);
        float3 primaryPosW = primarySd.posW;

        PathReservoir subpathReservoir;

        float3 res = 0.f;

        int currentPathTreeLength = 0;
        int currentRcVertexLength = 0;
        int prefixLength = 0;

        if (!restirpt.subpathSettings.directlyOutputColor)
        {
            subpathReservoir = {};
            subpathReservoir.init();
        }

        bool hasValidTemporalNeighbor = false;

        if (centralPrefixValid)
        {
            PathReservoir currentReservoir = restirpt.pathReservoirs[offset];
            currentPathTreeLength = currentReservoir.pathFlags.pathTreeLength();
            prefixLength = currentReservoir.pathFlags.prefixLength();
            currentRcVertexLength = currentReservoir.pathFlags.rcVertexLength();

            const int numNeighbors = restirpt.subpathSettings.finalGatherSuffixCount;

            float canonicalMisWeights = 0.f;
            int validNeighbors = 0;
            const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

            for (int neighborId = 0; neighborId < numNeighbors; neighborId++)
            {
                int2 neighborPixel = unpackSearchedPixel(temporalNeighborPixels[numNeighbors * offset + neighborId]);
                int neighborOffset = params.getReservoirOffset(neighborPixel);

                PathReservoir neighborReservoir = prevReservoirs[neighborOffset];
                if (!neighborValidMask[offset].isValid(neighborId)) continue;
                hasValidTemporalNeighbor = true;

                validNeighbors++;

                adjustNeighborWeight(neighborReservoir.M, numNeighbors);

                if (restirpt.subpathSettings.useTalbotMISForGather)
                {
                    // talbot MIS
                    float talbotMISWeightSum = 0.f;
                    int bufferJ = 0;
                    for (int j = 0; j < numNeighbors; j++)
                    {
                        if (neighborId == j) continue;

                        int2 dstPixel = unpackSearchedPixel(temporalNeighborPixels[numNeighbors * offset + j]);
                        int dstOffset = params.getReservoirOffset(dstPixel);

                        if (!neighborValidMask[offset].isValid(j)) continue;

                        ReconnectionData rcDataShift;
                        if (neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength() 
                        && ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
                            rcDataShift = reconnectionDataBuffer[rcBufferOffsets[numNeighbors * (numNeighbors+1) * offset + neighborId*numNeighbors+bufferJ]];
                        else
                            rcDataShift = fillReconnectionPartialShiftRcDataPrefix(dstOffset);

                        PathReservoir dstReservoir = prevReservoirs[dstOffset];
                        adjustNeighborWeight(dstReservoir.M, numNeighbors);
                        talbotMISWeightSum += computeTalbotMISWeightTerm(neighborReservoir, dstReservoir, rcDataShift);
                        bufferJ++;
                    }

                    ReconnectionData rcDataShift;
                    if (neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength() 
                    && ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
                        rcDataShift = reconnectionDataBuffer[rcBufferOffsets[numNeighbors * (numNeighbors+1) * offset + neighborId*numNeighbors+numNeighbors-1]];
                    else
                    {
                        PrefixGBuffer pg = restirpt.prefixGBuffer[offset];
                        rcDataShift = ReconnectionData(pg.hit, pg.wo, float3(1.f));
                    }
                    if (!restirpt.subpathSettings.directlyOutputColor)
                    {
                        StreamNeighborPathIntoPathReservoirTalbotMIS(subpathReservoir,
                                                                     currentReservoir,
                                                                     neighborReservoir,
                                                                     rcDataShift,
                                                                     talbotMISWeightSum,
                                                                     sg);
                    }
                    else
                    {
                        res += integrateWithNeighborReservoirSampleTalbotMIS(currentReservoir,
                                                                      neighborReservoir,
                                                                      rcDataShift,
                                                                      talbotMISWeightSum);
                    }
                }
                else
                {
                    ReconnectionData rcDataBackShift;
                    if (currentReservoir.pathFlags.rcVertexLength() > 1 + currentReservoir.pathFlags.prefixLength() 
                    && ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
                    {
                        rcDataBackShift = reconnectionDataBuffer[rcBufferOffsets[2 * numNeighbors * offset + 2 * neighborId]];
                    }
                    else
                    {
                        rcDataBackShift = fillReconnectionPartialShiftRcDataPrefix(neighborOffset);
                    }

                    ReconnectionData rcDataShift;
                    if (neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength() 
                    && ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
                    {
                        rcDataShift = reconnectionDataBuffer[rcBufferOffsets[2 * numNeighbors * offset + 2 * neighborId + 1]];
                    }
                    else
                    {
                        PrefixGBuffer pg = restirpt.prefixGBuffer[offset];
                        rcDataShift = ReconnectionData(pg.hit, pg.wo, float3(1.f));
                    }

                    if (!restirpt.subpathSettings.directlyOutputColor)
                    {
                        StreamNeighborPathIntoPathReservoir(subpathReservoir, currentReservoir, neighborReservoir, rcDataShift, rcDataBackShift,
                                                            1.f, canonicalMisWeights, sg);
                    }
                    else
                    {
                        res += integrateWithNeighborReservoirSample(currentReservoir, neighborReservoir, rcDataShift, rcDataBackShift,
                                                             1.f, canonicalMisWeights);
                    }
                }
            }

            canonicalMisWeights = hasValidTemporalNeighbor ? canonicalMisWeights : 1.f;
            validNeighbors = hasValidTemporalNeighbor ? validNeighbors : 1;

            if (hasCanonicalSuffix && !restirpt.subpathSettings.disableCanonical)
            {
                if (restirpt.subpathSettings.useTalbotMISForGather)
                {
                    // talbot MIS
                    float talbotMISWeightSum = 0.f;
                    for (int j = 0; j < numNeighbors; j++)
                    {
                        int2 dstPixel = unpackSearchedPixel(temporalNeighborPixels[numNeighbors * offset + j]);
                        int dstOffset = params.getReservoirOffset(dstPixel);
                        if (!neighborValidMask[offset].isValid(j)) continue;

                        PathReservoir dstReservoir = prevReservoirs[dstOffset];
                        ReconnectionData rcDataShift;
                        if (currentRcVertexLength > 1 + prefixLength 
                        && ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
                            rcDataShift = reconnectionDataBuffer[rcBufferOffsets[numNeighbors * (numNeighbors+1) * offset + numNeighbors*numNeighbors+j]];
                        else
                            rcDataShift = fillReconnectionPartialShiftRcDataPrefix(dstOffset);

                        adjustNeighborWeight(dstReservoir.M, numNeighbors);
                        talbotMISWeightSum += computeTalbotMISWeightTerm(currentReservoir, dstReservoir, rcDataShift);
                    }

                    canonicalMisWeights = currentReservoir.M * PathRisState::toScalar(currentReservoir.integrand) /
                                          (currentReservoir.M * PathRisState::toScalar(currentReservoir.integrand) + talbotMISWeightSum);
                    canonicalMisWeights = (isinf(canonicalMisWeights) || isnan(canonicalMisWeights)) ? 0.f : canonicalMisWeights;
                }
                if (!restirpt.subpathSettings.directlyOutputColor)
                {
                    subpathReservoir.merge(currentReservoir.integrand, 1.f, currentReservoir, sg, // multiply by M to compensate for russian roulette
                                           canonicalMisWeights * (!restirpt.subpathSettings.generateCanonicalSuffixForEachPrefix ? 
                                            restirpt.subpathSettings.numIntegrationPrefixes : 1.f));
                }
                else
                {
                    res += currentReservoir.integrand * currentReservoir.weight * canonicalMisWeights * (!restirpt.subpathSettings.generateCanonicalSuffixForEachPrefix ? 
                                            restirpt.subpathSettings.numIntegrationPrefixes : 1.f);
                }
            }

            if (!restirpt.subpathSettings.directlyOutputColor)
            {
                subpathReservoir.finalizeGRIS();
                subpathReservoir.initRandomSeed = currentReservoir.initRandomSeed; // this has to be the same as the new sample
                subpathReservoir.pathFlags.insertPathTreeLength(currentReservoir.pathFlags.pathTreeLength());
                subpathReservoir.pathFlags.insertPrefixLength(currentReservoir.pathFlags.prefixLength());
                if (!restirpt.subpathSettings.useTalbotMISForGather)
                    subpathReservoir.weight /= validNeighbors; // compensate for the fact that pairwise resampling MIS was not divided by k
                subpathReservoir.weight = (isnan(subpathReservoir.weight) || isinf(subpathReservoir.weight))
                                            ? 0.f
                                            : max(0.f, subpathReservoir.weight);
            }
            else
            {
                if (!restirpt.subpathSettings.useTalbotMISForGather)
                    res /= validNeighbors;
                res = any(isnan(res) || isinf(res))
                          ? 0.f
                          : max(0.f, res);
            }
        }

        int vertexBufferOffset = (kMaxSurfaceBounces + 1) * offset;

        float3 suffixIntegrand;
        if (!restirpt.subpathSettings.directlyOutputColor)
            suffixIntegrand = subpathReservoir.integrand;
        
        prefixLength = !centralPrefixValid ? currentPathTreeLength + 1 : prefixLength;  
        
        // multiply with prefix throughput
        if (centralPrefixValid)
        {
            if (prefixLength > 0)
            { 
                float3 prefixThroughput = restirpt.prefixThroughputs[offset].thp;

                if (!restirpt.subpathSettings.directlyOutputColor)
                {
                    subpathReservoir.integrand *= prefixThroughput;
                    subpathReservoir.weight *= restirpt.prefixThroughputs[offset].W;
                }
                else
                {
                    res *= prefixThroughput;
                }
            }
        }
        else
        {
            subpathReservoir.weight = 0;
            subpathReservoir.M = 1;
        }

        if (prefixLength > 0 && subpathReservoir.weight > 0.f)
        {
            // if we pick suffix path, may need to move rcVertex
            if (centralPrefixValid && !restirpt.subpathSettings.directlyOutputColor)
            {
                int prefixPartRcVertexLength = restirpt.prefixThroughputs[offset].getRcVertexLength();

                if (prefixPartRcVertexLength < prefixLength)
                {
                    // fill rc data here
                    subpathReservoir.rcHit = restirpt.prefixThroughputs[offset].rcHit;
                    subpathReservoir.rcIrrad = restirpt.prefixThroughputs[offset].rcThroughput * suffixIntegrand;
                    subpathReservoir.rcJacobian = restirpt.prefixThroughputs[offset].rcJacobian;
                    subpathReservoir.rcWi = restirpt.prefixThroughputs[offset].rcWi;
                    subpathReservoir.pathFlags.insertRcVertexLength(prefixPartRcVertexLength);
                    uint prefixRcFlags = restirpt.prefixThroughputs[offset].flags;
                    subpathReservoir.pathFlags.transferDeltaTransmissionBSDFEvent(prefixRcFlags, true);
                    subpathReservoir.pathFlags.transferDeltaTransmissionBSDFEvent(prefixRcFlags, false);
                }
                else
                {
                    PrefixGBuffer sfPrevVertex = restirpt.prefixGBuffer[offset];


                    ShadingData sfPrevSd = loadShadingData(sfPrevVertex.hit, 0.f, -sfPrevVertex.wo, false, gPathTracer.createTextureSampler(), false);
                    float3 sfPrevVertexPos = sfPrevSd.posW;
                    float3 sfPrevVertexN = sfPrevSd.faceN;


                    float prefixJacobian = restirpt.prefixThroughputs[offset].prefixJacobian; // sfPrevVertex.pdf * G_prevprev_prev
                    LastVertexState sfPrevFlags = LastVertexState(restirpt.prefixThroughputs[offset].getLastVertexStateFlags());

                    // sprcVertex is the first vertex of suffix, we look at whether the last vertex of prefix can be the rcVertex
                    if (subpathReservoir.pathFlags.rcVertexLength() == prefixLength + 1)
                    {
                        // note that rcVertexLength >= prefixLength is automatically true
                        bool canConnect = gPathTracer.restirpt.shiftMapping == (uint)ReSTIRPT::ShiftMapping::Reconnection && prefixLength == 1 ||
                                        gPathTracer.restirpt.shiftMapping == (uint)ReSTIRPT::ShiftMapping::Hybrid &&
                        // in non-immediate-NEE case, since the vertex is selected as suffix rc vertex, the previous one is guaranteed to be rough
                                            (!(subpathReservoir.pathFlags.lastVertexNEE() && subpathReservoir.pathFlags.pathLength() == prefixLength) || sfPrevFlags.isVertexClassifiedAsRoughForNEE()) 
                                            && sfPrevFlags.isLastVertexRough() && sfPrevFlags.isCurrentVertexFarFromPrev();
                        if (canConnect)
                        {

                            ShadingData sfFirstSd = loadShadingData(subpathReservoir.rcHit, sfPrevVertexPos, 0.f, false, gPathTracer.createTextureSampler(), false);
                            subpathReservoir.rcHit = sfPrevVertex.hit;

                            if (subpathReservoir.pathFlags.pathLength() > prefixLength)
                            {
                                float3 rcWi = subpathReservoir.rcWi;
                                const bool isRcVertexConnectedtoLight = subpathReservoir.pathFlags.pathLength() == prefixLength + 1;
                                bool isRcVertexNEE = subpathReservoir.pathFlags.lastVertexNEE() && isRcVertexConnectedtoLight;

                                const uint componentType2 = subpathReservoir.pathFlags.decodeBSDFComponent(false);
                                float dstPDF2BSDF, dstPDF2ForMIS;

                                const IMaterialInstance rcMI = gScene.materials.getMaterialInstance( sfFirstSd, gPathTracer.createTextureSampler(), getMaterialInstanceHintsAssumeTriangle(false));
                                float3 dstF2 = rcMI.evalBsdfAndPdf(sfFirstSd, rcWi, sg, isRcVertexNEE ? 0xf : 1 << componentType2, dstPDF2BSDF, dstPDF2ForMIS);


                                float MISForLightVertex = (kUseMIS && isRcVertexConnectedtoLight && subpathReservoir.pathFlags.lightType() != (uint)PathTracer::LightType::Analytic) ? 
                                PathTracer::evalMIS(1, isRcVertexNEE ? subpathReservoir.lightPdf : dstPDF2ForMIS, 1, isRcVertexNEE ? dstPDF2ForMIS : subpathReservoir.lightPdf) : 1;
                                float dstPDF2 = isRcVertexNEE ? subpathReservoir.lightPdf : dstPDF2BSDF;
                                // if don't need second part, impossible for isRcVertexConnectedtoLight to be true
                                float3 localWeight = MISForLightVertex * dstF2 / dstPDF2;

                                subpathReservoir.rcIrrad *= localWeight;
                            }
                            // else, suffix rcVertex is a light

                            // are we envmap?
                            if (!(subpathReservoir.pathFlags.pathLength() == prefixLength && subpathReservoir.pathFlags.lightType() == (uint)PathTracer::LightType::EnvMap))
                            {

                                subpathReservoir.rcWi = normalize(sfFirstSd.posW - sfPrevVertexPos);
                                float G_sfprev_sff = GeometryTerm(sfPrevVertexPos, sfFirstSd.posW, sfFirstSd.faceN);

                                subpathReservoir.lightPdf /= G_sfprev_sff;   // convert to solid angle measure
                                subpathReservoir.rcJacobian /= G_sfprev_sff; // sprcJacobian was p1*G in BSDF path case, divide by G we get p1
                            }

                            float pdfAfterNewRc = subpathReservoir.pathFlags.lastVertexNEE() && subpathReservoir.pathFlags.pathLength() == prefixLength ? subpathReservoir.lightPdf : subpathReservoir.rcJacobian;
                            if (subpathReservoir.pathFlags.pathLength() > prefixLength) // we need to recompute p1 since sprcJacobian was p1*G*p2
                            {

                                IMaterialInstance sfPrevMi = gScene.materials.getMaterialInstance(sfPrevSd, gPathTracer.createTextureSampler());
                                sfPrevSd.mtl.setLobeMask(1 << subpathReservoir.pathFlags.decodeBSDFComponent(true));
                                pdfAfterNewRc = sfPrevMi.evalPdf(sfPrevSd, subpathReservoir.rcWi);

                            }
                            subpathReservoir.rcJacobian = prefixJacobian * pdfAfterNewRc;
                            subpathReservoir.pathFlags.insertRcVertexLength(prefixLength);
                            subpathReservoir.pathFlags.insertIsDeltaEvent(subpathReservoir.pathFlags.decodeIsDeltaEvent(true), false);
                            subpathReservoir.pathFlags.insertIsTransmissionEvent(subpathReservoir.pathFlags.decodeIsTransmissionEvent(true), false);
                            subpathReservoir.pathFlags.insertBSDFComponentType(subpathReservoir.pathFlags.decodeBSDFComponent(true), false);
                            subpathReservoir.pathFlags.transferDeltaTransmissionBSDFEvent(sfPrevFlags, true);
                        }
                        // otherwise, suffix rcVertex is rcVertex
                    }
                    else
                    {
                        BSDFSample bs;
                        SampleGenerator tempSg = SampleGenerator(subpathReservoir.suffixInitRandomSeed);
                        
                        IMaterialInstance sfPrevMi = gScene.materials.getMaterialInstance(sfPrevSd, gPathTracer.createTextureSampler());
                        sfPrevSd.mtl.setLobeMask(1);
                        sfPrevMi.sample(sfPrevSd, tempSg, bs);
                        BSDFProperties bsdfProperties = sfPrevMi.getProperties(sfPrevSd);

                        uint lobeType = decodeLobeType(bs);

                        bool isCurrentBounceRough = lobeType <= 1 || !bs.isLobe(LobeType::Delta) && bsdfProperties.roughness > restirpt.settings.specularRoughnessThreshold;
                        
                        bool canConnect = gPathTracer.restirpt.shiftMapping == (uint)ReSTIRPT::ShiftMapping::Reconnection && prefixLength == 1 ||
                                        gPathTracer.restirpt.shiftMapping == (uint)ReSTIRPT::ShiftMapping::Hybrid &&
                                            sfPrevFlags.isLastVertexRough() && sfPrevFlags.isCurrentVertexFarFromPrev() && isCurrentBounceRough;
                        if (canConnect)
                        {
                            float3 rcIrrad = 0.f;
                            if (subpathReservoir.pathFlags.pathLength() == prefixLength) // suffix only has one vertex, but it is not a sprcVertex (BSDF path)
                            {
                                if (subpathReservoir.pathFlags.lightType() == (uint)PathTracer::LightType::EnvMap)
                                {
                                    rcIrrad = gPathTracer.envMapSampler.eval(bs.wo, 0.f);
                                }
                                else
                                {
                                    // in this case, suffix rcVertex length is kMaxRcLength
                                    // we can't use the "suffixIntegrand / bs.weight" divide because there is an unknown MIS weight left
                                    subpathReservoir.rcHit.setTriangle(); // was marked invalid in PathTracer.slang
                                    HitInfo lightHitInfo = subpathReservoir.rcHit;
                                    ShadingData lightSd = loadShadingData(lightHitInfo, sfPrevSd.posW, 0.f, false, gPathTracer.createTextureSampler(), false);
                                    rcIrrad = fetchLightEmission(lightSd);
                                    
                                }
                            }
                            else
                            {
                                rcIrrad = suffixIntegrand / bs.weight;
                                if (isnan(rcIrrad.x)) rcIrrad.x = 0.f;
                                if (isnan(rcIrrad.y)) rcIrrad.y = 0.f;
                                if (isnan(rcIrrad.z)) rcIrrad.z = 0.f;
                            }

                            subpathReservoir.rcHit = sfPrevVertex.hit;
                            subpathReservoir.rcIrrad = rcIrrad;
                            subpathReservoir.rcWi = bs.wo;
                            subpathReservoir.rcJacobian = prefixJacobian * bs.pdf;
                            subpathReservoir.pathFlags.insertRcVertexLength(prefixLength);
                            subpathReservoir.pathFlags.insertIsDeltaEvent(false, false);
                            subpathReservoir.pathFlags.insertIsTransmissionEvent(bs.isLobe(LobeType::Transmission), false);
                            subpathReservoir.pathFlags.insertBSDFComponentType(lobeType, false);
                            subpathReservoir.pathFlags.transferDeltaTransmissionBSDFEvent(sfPrevFlags, true);
                        }
                        // otherwise, suffix rcVertex is rcVertex
                    }
                }
            }
        }

        subpathReservoir.weight = (isnan(subpathReservoir.weight) || isinf(subpathReservoir.weight))
                                      ? 0.f
                                      : max(0.f, subpathReservoir.weight);
        subpathReservoir.resetM();

        subpathReservoir.pathFlags.insertPathTreeLength(currentPathTreeLength);
        subpathReservoir.pathFlags.insertPrefixLength(prefixLength);

        // merge with reservoir

        if (restirpt.subpathSettings.numIntegrationPrefixes > 1)
        {
            if (restirpt.subpathSettings.directlyOutputColor)
            {
                // add prefix contribution
                if (integrationPrefixId == restirpt.subpathSettings.numIntegrationPrefixes - 1)
                {
                    res += restirpt.prefixPathReservoirs[offset].integrand;
                }
                outColor[pixel] += float4(res / restirpt.subpathSettings.numIntegrationPrefixes, 0.f);
            }
            else
            {
                if (!restirpt.subpathSettings.directlyOutputColor)
                {
                    PathReservoir finalReservoir;
                    if (integrationPrefixId == 0)
                        finalReservoir.init();
                    else
                        finalReservoir = reservoirs[offset];
                    finalReservoir.beginGRIS(1.f);
                    finalReservoir.mergeSameDomain(subpathReservoir, sg, 1.f);
                    finalReservoir.finalizeGRIS();

                    if (integrationPrefixId == restirpt.subpathSettings.numIntegrationPrefixes - 1)
                    {
                        finalReservoir.beginGRIS(1.f);
                        finalReservoir.mergeSameDomain(restirpt.prefixPathReservoirs[offset], sg, 1.f);
                        finalReservoir.M = restirpt.subpathSettings.numIntegrationPrefixes;
                        finalReservoir.finalizeGRISDivideByM();
                        finalReservoir.M = 1.f;
                    }
                    reservoirs[offset] = finalReservoir;
                } 
            }
        }
        else
        {
            // merge with preifx reservoir
            if (integrationPrefixId == restirpt.subpathSettings.numIntegrationPrefixes - 1)
            {
                if (!restirpt.subpathSettings.directlyOutputColor)
                {
                    subpathReservoir.beginGRIS(1.f);
                    bool selectPrefix = subpathReservoir.mergeSameDomain(restirpt.prefixPathReservoirs[offset], sg, 1.f);
                    subpathReservoir.finalizeGRIS();
                }
                else
                {
                    res += restirpt.prefixPathReservoirs[offset].integrand;
                }
            }

            if (restirpt.subpathSettings.directlyOutputColor)
            {
                outColor[pixel] += float4(res, 0.f);
            }
            else
            {
                reservoirs[offset] = subpathReservoir;
            }
        }
    }

    void runTemporal(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
    {
        int2 pixel = dispatchThreadId.xy;
        const int offset = params.getReservoirOffset(pixel);

        printSetPixel(pixel);
        logSetPixel(pixel);

        bool insideScreen = isValidScreenRegion(params, pixel);
        if (!insideScreen) return;
        bool centralPrefixValid = isPixelPrefixLastSdValid(offset);
        if (!centralPrefixValid) return;

        SampleGenerator sg = SampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.suffixResamplingSgOffset() + suffixReuseRoundId);

        PathReservoir subpathReservoir;
        subpathReservoir.init();

        int vertexBufferOffset = (kMaxSurfaceBounces + 1) * offset;

        float3 color = 0.f;

        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

        PathReservoir currentReservoir = reservoirs[offset];

        int currentPathTreeLength = currentReservoir.pathFlags.pathTreeLength();
        int currentPrefixLength = currentReservoir.pathFlags.prefixLength();
        bool currentUserFlagSet = currentReservoir.pathFlags.isUserFlagSet();

        static const bool doTemporalUpdate = restirpt.kTemporalUpdateForDynamicScene;

        bool hasValidTemporalNeighbor = false;

        int numNeighbors = 1;

        float2 motionVector = motionVectors[pixel];
        int2 reprojPixel = pixel + motionVector * params.frameDim + 0.5;
        float canonicalMisWeights = 0.f;

        float3 currentPrefixDir = prefixGBuffer[offset].wo;

        const PrevCameraFrame pcf = { prevCameraU, prevCameraV, prevCameraW, prevJitterX, prevJitterY };

        ReconnectionData rcDataCenter = fillReconnectionPartialShiftRcDataPrefix(offset);

        for (int neighborId = 0; neighborId < numNeighbors; neighborId++)
        {
            int2 neighborPixel = getNextNeighborPixel(startIndex, reprojPixel, neighborId - 1);

            if (!neighborValidMask[offset].isValid(neighborId)) continue;

            int neighborOffset = params.getReservoirOffset(neighborPixel);

            PathReservoir neighborReservoir = prevReservoirs[neighborOffset];

            neighborReservoir.M = min(restirpt.subpathSettings.temporalHistoryLength, neighborReservoir.M);
            hasValidTemporalNeighbor = true;

            ReconnectionData rcDataBackShift;

            if (currentReservoir.pathFlags.rcVertexLength() > 1 + currentReservoir.pathFlags.prefixLength()
                && ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
            {
                rcDataBackShift = reconnectionDataBuffer[rcBufferOffsets[2 * numNeighbors * offset + 2 * neighborId]];
            }
            else
            {
                rcDataBackShift = fillReconnectionPartialShiftRcDataPrefixPrevFrame(neighborOffset);
            }

            ReconnectionData rcDataShift;

            if (neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength()
                && ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
            {
                rcDataShift = reconnectionDataBuffer[rcBufferOffsets[2 * numNeighbors * offset + 2 * neighborId + 1]];
            }
            else
            {
                rcDataShift = rcDataCenter;
            }

            bool selected = StreamNeighborPathIntoPathReservoir(subpathReservoir, currentReservoir, neighborReservoir, rcDataShift, rcDataBackShift,
                                                                1.f, 
                                                                canonicalMisWeights, sg, true, doTemporalUpdate);

        };

        canonicalMisWeights = hasValidTemporalNeighbor ? canonicalMisWeights : 1.f;

        bool selected = subpathReservoir.merge(currentReservoir.integrand, 1.f, currentReservoir, sg, canonicalMisWeights);

        subpathReservoir.finalizeGRIS();
        subpathReservoir.weight = (isnan(subpathReservoir.weight) || isinf(subpathReservoir.weight))
                                      ? 0.f
                                      : max(0.f, subpathReservoir.weight);

        subpathReservoir.pathFlags.insertPathTreeLength(currentPathTreeLength); 
        subpathReservoir.pathFlags.insertPrefixLength(currentPrefixLength);

        subpathReservoir.initRandomSeed = currentReservoir.initRandomSeed;

        reservoirs[offset] = subpathReservoir;
    }

    void runSpatial(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
    {
        int2 pixel = dispatchThreadId.xy;
        const int offset = params.getReservoirOffset(pixel);

        printSetPixel(pixel);
        logSetPixel(pixel);

        bool insideScreen = isValidScreenRegion(params, pixel);
        if (!insideScreen) return;
        bool centralPrefixValid = isPixelPrefixLastSdValid(offset);
        if (!centralPrefixValid) return;

        SampleGenerator sg = SampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.suffixResamplingSgOffset() + suffixReuseRoundId);

        PathReservoir subpathReservoir;
        subpathReservoir.init();

        int vertexBufferOffset = (kMaxSurfaceBounces + 1) * offset;

        float3 color = 0.f;
        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

        PathReservoir currentReservoir = prevReservoirs[offset];

        int currentPathTreeLength = currentReservoir.pathFlags.pathTreeLength();
        int currentPrefixLength = currentReservoir.pathFlags.prefixLength();
        bool currentUserFlagSet = currentReservoir.pathFlags.isUserFlagSet();

        int validNeighbors = 0;

        int numNeighbors = restirpt.subpathSettings.suffixSpatialNeighborCount;
        int2 reprojPixel = pixel;

        float canonicalMisWeights = 1.f;

        ReconnectionData rcDataCenter = fillReconnectionPartialShiftRcDataPrefix(offset);

        for (int neighborId = 0; neighborId < numNeighbors; neighborId++)
        {
            int2 neighborPixel = getNextNeighborPixel(startIndex, reprojPixel, neighborId);

            if (!neighborValidMask[offset].isValid(neighborId)) continue;

            int neighborOffset = params.getReservoirOffset(neighborPixel);

            validNeighbors++;

            PathReservoir neighborReservoir = prevReservoirs[neighborOffset];

            ReconnectionData rcDataBackShift;

            if (currentReservoir.pathFlags.rcVertexLength() > 1 + currentReservoir.pathFlags.prefixLength() 
                && ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
            {
                rcDataBackShift = reconnectionDataBuffer[rcBufferOffsets[2 * numNeighbors * offset + 2 * neighborId]];
            }
            else
            {
                rcDataBackShift = fillReconnectionPartialShiftRcDataPrefix(neighborOffset);
            }

            ReconnectionData rcDataShift;

            if (neighborReservoir.pathFlags.rcVertexLength() > 1 + neighborReservoir.pathFlags.prefixLength() 
                && ReSTIRPT::ShiftMapping(restirpt.shiftMapping) == ReSTIRPT::ShiftMapping::Hybrid)
            {
                rcDataShift = reconnectionDataBuffer[rcBufferOffsets[2 * numNeighbors * offset + 2 * neighborId + 1]];
            }
            else
            {
                rcDataShift = rcDataCenter;
            }

            bool selected = StreamNeighborPathIntoPathReservoir(subpathReservoir, currentReservoir, neighborReservoir, rcDataShift, rcDataBackShift,
                                                                numNeighbors, 
                                                                canonicalMisWeights, sg);
        };

        bool selected = subpathReservoir.merge(currentReservoir.integrand, 1.f, currentReservoir, sg, canonicalMisWeights);

        subpathReservoir.finalizeGRIS();
        subpathReservoir.weight /= (validNeighbors + 1);
        subpathReservoir.weight = (isnan(subpathReservoir.weight) || isinf(subpathReservoir.weight))
                                      ? 0.f
                                      : max(0.f, subpathReservoir.weight);

        subpathReservoir.pathFlags.insertPathTreeLength(currentPathTreeLength);
        subpathReservoir.pathFlags.insertPrefixLength(currentPrefixLength);
        subpathReservoir.initRandomSeed = currentReservoir.initRandomSeed;

        reservoirs[offset] = subpathReservoir;
    }
}

cbuffer CB
{
    SuffixResampling gSuffixResampling;
}

[numthreads(16, 16, 1)]
void spatial(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
{
    gSuffixResampling.runSpatial(dispatchThreadId, groupThreadIdx, groupID);
}

[numthreads(16, 16, 1)]
void temporal(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
{
    gSuffixResampling.runTemporal(dispatchThreadId, groupThreadIdx, groupID);
}

[numthreads(16, 16, 1)]
void gather(uint3 dispatchThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupID: SV_GroupID)
{
    gSuffixResampling.runGatherSuffixes(dispatchThreadId, groupThreadIdx, groupID);
}
