/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "ReSTIRPathTracing.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"
#include "Utils/Timing/Profiler.h"
#include "Utils/Color/ColorHelpers.slang"
#include "../../../RenderPasses/PathTracer/Params.slang"
#include "BoundingBoxAccelerationStructureBuilder.h"

namespace Falcor
{
    namespace
    {
        const char kReflectTypesFile[] = "Rendering/ReSTIRPathTracing/ReflectTypes.cs.slang";
        const char kSuffixSpatialResamplingFile[] = "Rendering/ReSTIRPathTracing/SuffixResampling.cs.slang";
        const char kSuffixTemporalResamplingFile[] = "Rendering/ReSTIRPathTracing/SuffixResampling.cs.slang";
        const char kSuffixResamplingFile[] = "Rendering/ReSTIRPathTracing/SuffixResampling.cs.slang";
        const char kTemporalResamplingFile[] = "Rendering/ReSTIRPathTracing/TemporalResampling.cs.slang";
        const char kSpatialResamplingFile[] = "Rendering/ReSTIRPathTracing/SpatialResampling.cs.slang";
        const char kTemporalRetraceFile[] = "Rendering/ReSTIRPathTracing/TemporalPathRetrace.cs.slang";
        const char kSpatialRetraceFile[] = "Rendering/ReSTIRPathTracing/SpatialPathRetrace.cs.slang";
        const char kProduceRetraceWorkload[] = "Rendering/ReSTIRPathTracing/ProduceRetraceWorkload.cs.slang";
        const char kSuffixRetraceFile[] = "Rendering/ReSTIRPathTracing/SuffixPathRetrace.cs.slang";
        const char kSuffixProduceRetraceWorkload[] = "Rendering/ReSTIRPathTracing/SuffixProduceRetraceWorkload.cs.slang";
        const char kSuffixRetraceTalbotFile[] = "Rendering/ReSTIRPathTracing/SuffixPathRetraceTalbot.cs.slang";
        const char kSuffixProduceRetraceTalbotWorkload[] = "Rendering/ReSTIRPathTracing/SuffixProduceRetraceTalbotWorkload.cs.slang";

        const char kPrefixRetraceFile[] = "Rendering/ReSTIRPathTracing/PrefixPathRetrace.cs.slang";
        const char kPrefixProduceRetraceWorkload[] = "Rendering/ReSTIRPathTracing/PrefixProduceRetraceWorkload.cs.slang";
        const char kPrefixResampling[] = "Rendering/ReSTIRPathTracing/PrefixResampling.cs.slang";

        const char kTraceNewSuffixes[] = "Rendering/ReSTIRPathTracing/TraceNewSuffixes.cs.slang";
        const char kPrefixNeighborSearch[] = "Rendering/ReSTIRPathTracing/PrefixNeighborSearch.cs.slang";
        const char kTraceNewPrefixes[] = "Rendering/ReSTIRPathTracing/TraceNewPrefixes.cs.slang";

        const ShaderModel kShaderModel = ShaderModel::SM6_5;

        const Gui::DropdownList kDebugOutputList =
        {
            { (uint32_t)ReSTIRPathTracing::DebugOutput::Disabled, "Disabled" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::Position, "Position" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::Depth, "Depth" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::Normal, "Normal" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::FaceNormal, "FaceNormal" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::DiffuseWeight, "DiffuseWeight" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::SpecularWeight, "SpecularWeight" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::SpecularRoughness, "SpecularRoughness" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::PackedNormal, "PackedNormal" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::PackedDepth, "PackedDepth" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::InitialWeight, "InitialWeight" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::TemporalReuse, "TemporalReuse" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::SpatialReuse, "SpatialReuse" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::FinalSampleDir, "FinalSampleDir" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::FinalSampleDistance, "FinalSampleDistance" },
            { (uint32_t)ReSTIRPathTracing::DebugOutput::FinalSampleLi, "FinalSampleLi" },
        };

        const Gui::DropdownList kShiftMappingList =
        {
            { (uint32_t)ReSTIRPT::ShiftMapping::Reconnection, "Reconnection" },
            { (uint32_t)ReSTIRPT::ShiftMapping::Hybrid, "Hybrid" },
        };

        const Gui::DropdownList kRetraceScheduleType =
        {
            { (uint32_t)ReSTIRPT::RetraceScheduleType::Naive, "Naive" },
            { (uint32_t)ReSTIRPT::RetraceScheduleType::Compact, "Compact" },
        };

        const Gui::DropdownList kAdaptivePrefixType =
        {
            { 0, "After First Diffuse" },
            { 1, "First Diffuse" },
        };

        const Gui::DropdownList kAmortizeLengthList = {
            {4, "4x4"},
            {8, "8x8"},
            {16, "16x16"},
        };

        const Gui::DropdownList kGRISTileLengthList = {
            {4, "4x4"},
            {8, "8x8"},
            {16, "16x16"},
        };

        const Gui::DropdownList kFireflyShareBlockLen = {
            {32, "32x32"},
            {64, "64x64"},
            {128, "128x128"},
            {256, "256x256"},
            {512, "512x512"},
        };

        const Gui::DropdownList kSubpathMISKind = {
            {(uint32_t)ReSTIRPT::SubpathMIS::Binary, "Binary"},
            {(uint32_t)ReSTIRPT::SubpathMIS::P, "P"},
            {(uint32_t)ReSTIRPT::SubpathMIS::PHat, "PHat"},
        };

        const Gui::DropdownList kKNNAdaptiveRadiusType = {
            {(uint32_t)ReSTIRPT::KNNAdaptiveRadiusType::NonAdaptive, "NonAdaptive"},
            {(uint32_t)ReSTIRPT::KNNAdaptiveRadiusType::RayCone, "RayCone"},
        };

        const uint32_t kNeighborOffsetCount = 8192;
    }


    void ReSTIRPathTracing::createComputePass(ref<ComputePass>& pPass, std::string shaderFile, DefineList defines, ProgramDesc baseDesc, std::string entryFunction)
    {
        if (!pPass)
        {
            ProgramDesc desc = baseDesc;
            desc.addShaderLibrary(shaderFile).csEntry(entryFunction == "" ? "main" : entryFunction);
            pPass = ComputePass::create(mpDevice, desc, defines, false);
        }
        pPass->getProgram()->addDefines(defines);
        pPass->setVars(nullptr);
    }


    ReSTIRPathTracing::ReSTIRPathTracing(const ref<Scene>& pScene, const DefineList& ownerDefines, const Options& options, const std::shared_ptr<PixelStats>& pPixelStats)
        : mpScene(pScene),
        mpDevice(pScene->getDevice()),
        mOptions(options)
    {
        FALCOR_ASSERT(mpScene);

        mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);

        // Create compute pass for reflecting data types.
        ProgramDesc desc;
        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(ownerDefines);
        desc.addShaderLibrary(kReflectTypesFile).csEntry("main").setShaderModel(kShaderModel);
        mpReflectTypes = ComputePass::create(mpDevice,desc, defines);

        // Create neighbor offset texture.
        mpNeighborOffsets = createNeighborOffsetTexture(kNeighborOffsetCount);

        mpPixelStats = pPixelStats;
    }

    DefineList ReSTIRPathTracing::getDefines() const
    {
        DefineList defines;
        defines.add("TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE", mOptions.temporalUpdateForDynamicScene ? "1": "0");
        defines.add("USE_RESERVOIR_COMPRESSION", mOptions.useReservoirCompression ? "1" : "0");
        defines.add("RETRACE_SCHEDULE_TYPE", std::to_string((uint32_t)mOptions.retraceScheduleType));
        defines.add("COMPRESS_PREFIX_SEARCH_ENTRY", mOptions.subpathSetting.compressNeighborSearchKey ? "1" : "0");
        defines.add("USE_PREV_FRAME_SCENE_DATA", mOptions.usePrevFrameSceneData ? "1" : "0");

        return defines;
    }

    void ReSTIRPathTracing::setShaderData(const ShaderVar& var) const
    {
        var["settings"]["localStrategyType"] = mOptions.shiftMappingSettings.localStrategyType;
        var["settings"]["jacobianRejectionThreshold"] = mOptions.shiftMappingSettings.jacobianRejectionThreshold;
        var["settings"]["specularRoughnessThreshold"] = mOptions.shiftMappingSettings.specularRoughnessThreshold;
        var["settings"]["nearFieldDistanceThreshold"] = mOptions.shiftMappingSettings.nearFieldDistanceThreshold;
        var["settings"]["rejectShiftBasedOnJacobian"] = mOptions.shiftMappingSettings.rejectShiftBasedOnJacobian;

        var["subpathSettings"]["directlyOutputColor"] = mOptions.subpathSetting.directlyOutputColor;
        var["subpathSettings"]["adaptivePrefixLength"] = mOptions.subpathSetting.adaptivePrefixLength;
        var["subpathSettings"]["avoidSpecularPrefixEndVertex"] = mOptions.subpathSetting.avoidSpecularPrefixEndVertex;
        var["subpathSettings"]["avoidShortPrefixEndSegment"] = mOptions.subpathSetting.avoidShortPrefixEndSegment;
        var["subpathSettings"]["shortSegmentThreshold"] = mOptions.subpathSetting.shortSegmentThreshold;

        var["subpathSettings"]["subpathMISKind"] = mOptions.subpathSetting.subpathMISKind;
        var["subpathSettings"]["suffixSpatialNeighborCount"] = mOptions.subpathSetting.suffixSpatialNeighborCount;
        var["subpathSettings"]["suffixSpatialReuseRadius"] = mOptions.subpathSetting.suffixSpatialReuseRadius;
        var["subpathSettings"]["suffixSpatialReuseRounds"] = mOptions.subpathSetting.suffixSpatialReuseRounds;
        var["subpathSettings"]["numIntegrationPrefixes"] = mOptions.subpathSetting.numIntegrationPrefixes;
        var["subpathSettings"]["generateCanonicalSuffixForEachPrefix"] = mOptions.subpathSetting.generateCanonicalSuffixForEachPrefix;

        var["subpathSettings"]["suffixTemporalReuse"] = mOptions.subpathSetting.suffixTemporalReuse;
        var["subpathSettings"]["temporalHistoryLength"] = mOptions.subpathSetting.temporalHistoryLength;

        var["subpathSettings"]["prefixNeighborSearchRadius"] = mOptions.subpathSetting.prefixNeighborSearchRadius;
        var["subpathSettings"]["prefixNeighborSearchNeighborCount"] = mOptions.subpathSetting.prefixNeighborSearchNeighborCount;
        var["subpathSettings"]["finalGatherSuffixCount"] = mOptions.subpathSetting.finalGatherSuffixCount;

        var["subpathSettings"]["useTalbotMISForGather"] = mOptions.subpathSetting.useTalbotMISForGather;
        var["subpathSettings"]["nonCanonicalWeightMultiplier"] = mOptions.subpathSetting.nonCanonicalWeightMultiplier;
        var["subpathSettings"]["disableCanonical"] = mOptions.subpathSetting.disableCanonical;
        var["subpathSettings"]["compressNeighborSearchKey"] = mOptions.subpathSetting.compressNeighborSearchKey;


        var["subpathSettings"]["knnSearchRadiusMultiplier"] = mOptions.subpathSetting.knnSearchRadiusMultiplier;
        var["subpathSettings"]["knnSearchAdaptiveRadiusType"] = mOptions.subpathSetting.knnSearchAdaptiveRadiusType;
        var["subpathSettings"]["knnIncludeDirectionSearch"] = mOptions.subpathSetting.knnIncludeDirectionSearch;

        var["subpathSettings"]["useMMIS"] = mOptions.subpathSetting.useMMIS;

        var["numSpatialRounds"] = mOptions.spatialIterations;
        var["numAdditionalSpatialRounds"] = mOptions.additionalSpatialIterations;
        var["additionalSpatial"] = mOptions.useAdditionalSpatialResampling;

        var["minimumPrefixLength"] = mOptions.subpathReuse ? mOptions.minimumPrefixLength : 0;

        int numRounds = mOptions.subpathSetting.suffixSpatialReuseRounds + 1; //include the prefix streaming pass
        numRounds = mOptions.subpathSetting.suffixTemporalReuse ? numRounds + 1 : numRounds;
        var["suffixSpatialRounds"] = numRounds;
        var["pathReservoirs"] = mOptions.subpathReuse ? mpScratchReservoirs : mpReservoirs;
        var["prefixGBuffer"] = mpScratchPrefixGBuffer;
        var["prefixPathReservoirs"] = mpPrefixPathReservoirs;
        var["prefixThroughputs"] = mpPrefixThroughputs;
        var["prefixReservoirs"] = mpPrefixReservoirs;
        float3 worldBoundExtent = mpScene->getSceneBounds().extent();
        var["sceneRadius"] = std::min(worldBoundExtent.x, std::min(worldBoundExtent.y, worldBoundExtent.z));
        
        var["scratchVertices"] = mpScratchVerticesBuffer;
        var["needResetTemporalHistory"] = mResetTemporalReservoirs;
        var["samplesPerPixel"] = mPathTracerParams.samplesPerPixel;
        var["subpathReuse"] = mOptions.subpathReuse;
        var["shiftMapping"] = (uint32_t)mOptions.shiftMapping;
    }

    void ReSTIRPathTracing::setPathTracerParams(int useFixedSeed, uint fixedSeed,
    float lodBias, float specularRoughnessThreshold, uint2 frameDim, uint2 screenTiles, uint frameCount, uint seed,
    int samplesPerPixel, int DIMode)
    {
        mPathTracerParams.useFixedSeed = useFixedSeed;
        mPathTracerParams.fixedSeed = fixedSeed;
        mPathTracerParams.lodBias = lodBias;
        mPathTracerParams.specularRoughnessThreshold = specularRoughnessThreshold;
        mPathTracerParams.frameDim = frameDim;
        mPathTracerParams.screenTiles = screenTiles;
        mPathTracerParams.frameCount = frameCount;
        mPathTracerParams.seed = seed;
        mPathTracerParams.samplesPerPixel = samplesPerPixel;
        mPathTracerParams.DIMode = DIMode;
    }

    void ReSTIRPathTracing::setOwnerDefines(DefineList defines)
    {
        mOwnerDefines = defines;
    }

    void ReSTIRPathTracing::setSharedStaticParams(uint32_t samplesPerPixel, uint32_t maxSurfaceBounces, bool useNEE)
    {
        mStaticParams.maxSurfaceBounces = maxSurfaceBounces;
        mStaticParams.useNEE = useNEE;
    }

    void ReSTIRPathTracing::createPathTracerBlock()
    {
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("pathTracer");
        mpPathTracerBlock = ParameterBlock::create(mpDevice,reflector);
    }

    ref<ParameterBlock> ReSTIRPathTracing::getPathTracerBlock()
    {
        return mpPathTracerBlock;
    }

    void ReSTIRPathTracing::setReservoirData(const ShaderVar& var) const
    {
        var["pathReservoirs"] = mpReservoirs;
    }

    bool ReSTIRPathTracing::renderUI(Gui::Widgets& widget)
    {
        bool dirty = false;

        if (auto group = widget.group("Performance settings", true))
        {
            mReallocate |= group.checkbox("Use reservoir compression", mOptions.useReservoirCompression);
            mReallocate |= group.dropdown("Retrace Schedule Type", kRetraceScheduleType, reinterpret_cast<uint32_t&>(mOptions.retraceScheduleType));
        }

        if (auto group = widget.group("Subpath reuse", true))
        {
            mRecompile |= group.checkbox("Enable Subpath Reuse", mOptions.subpathReuse);

            if (mOptions.subpathReuse)
            {
                mReallocate |= group.checkbox("Directly Output Color", mOptions.subpathSetting.directlyOutputColor);
                dirty |= group.var("Num Integration Prefixes", mOptions.subpathSetting.numIntegrationPrefixes, 1, 128);
                dirty |= group.checkbox("Generate Canonical Suffix For Each Prefix", mOptions.subpathSetting.generateCanonicalSuffixForEachPrefix);
                dirty |= group.checkbox("Use MMIS", mOptions.subpathSetting.useMMIS);
                dirty |= group.var("Min Prefix Length", mOptions.minimumPrefixLength, 1u, mStaticParams.maxSurfaceBounces);
                dirty |= group.checkbox("Adaptive Prefix Length", mOptions.subpathSetting.adaptivePrefixLength);
                dirty |= group.checkbox("Avoid Specular Prefix End Vertex", mOptions.subpathSetting.avoidSpecularPrefixEndVertex);
                dirty |= group.checkbox("Avoid Short Prefix End Segment", mOptions.subpathSetting.avoidShortPrefixEndSegment);
                dirty |= group.var("Short Segment Threshold", mOptions.subpathSetting.shortSegmentThreshold, 0.f, 0.1f);

                dirty |= group.dropdown("Subpath MIS Kind", kSubpathMISKind, reinterpret_cast<uint32_t&>(mOptions.subpathSetting.subpathMISKind));
                mReallocate |= group.var("Suffix Spatial Neighbors", mOptions.subpathSetting.suffixSpatialNeighborCount, 1, 8);
                dirty |= group.var("Suffix Spatial Reuse Radius", mOptions.subpathSetting.suffixSpatialReuseRadius, 0.f, 100.f);

                {
                    dirty |= group.var("Suffix Reuse rounds", mOptions.subpathSetting.suffixSpatialReuseRounds, 0, 16);
                    dirty |= group.checkbox("Suffix Temporal Reuse", mOptions.subpathSetting.suffixTemporalReuse);
                    dirty |= group.var("Suffix Temopral History Length", mOptions.subpathSetting.temporalHistoryLength, 0, 100);

                    mReallocate |= group.var("Supporting Neighbors", mOptions.subpathSetting.finalGatherSuffixCount, 1, 8);
                    dirty |= group.var("Supporting Neighbor Search Radius", mOptions.subpathSetting.prefixNeighborSearchRadius, 0, 100);
                    dirty |= group.var("Supporting Neighbor Search Neighbors", mOptions.subpathSetting.prefixNeighborSearchNeighborCount, 0, 100);
                    mReallocate |= group.checkbox("Use Talbot MIS For Gather", mOptions.subpathSetting.useTalbotMISForGather);
                    dirty |= group.var("Non-Canonical Weight Multiplier", mOptions.subpathSetting.nonCanonicalWeightMultiplier, 0.f, 100.f);
                    dirty |= group.checkbox("Disable Canonical", mOptions.subpathSetting.disableCanonical);

                    dirty |= group.var("KNN Search Radius Multiplier", mOptions.subpathSetting.knnSearchRadiusMultiplier);
                    dirty |= group.dropdown("KNN Search Adaptive Type", kKNNAdaptiveRadiusType, reinterpret_cast<uint32_t&>(mOptions.subpathSetting.knnSearchAdaptiveRadiusType));
                    dirty |= group.checkbox("KNN Inlucde Direction Search For Low Roughness", mOptions.subpathSetting.knnIncludeDirectionSearch);

                    mReallocate |= group.checkbox("Compress Neighbor Search Key", mOptions.subpathSetting.compressNeighborSearchKey);
                }
            }
        }

        if (auto group = widget.group("Shift mapping options", true))
        {
            mRecompile |= group.dropdown("Shift Mapping", kShiftMappingList, reinterpret_cast<uint32_t&>(mOptions.shiftMapping));

            if (mOptions.shiftMapping == ReSTIRPT::ShiftMapping::Hybrid)
            {
                dirty |= group.var("Distance Threshold", mOptions.shiftMappingSettings.nearFieldDistanceThreshold);
                dirty |= group.var("Roughness Threshold", mOptions.shiftMappingSettings.specularRoughnessThreshold);
            }
        }

        if (auto group = widget.group("Temporal resampling", true))
        {
            dirty |= group.checkbox("Use temporal resampling", mOptions.useTemporalResampling);

            dirty |= group.var("Max history length", mOptions.maxHistoryLength, 0u, 100u);
            group.tooltip("Maximum temporal history length.");

            mReallocate |= group.checkbox("Temporal Reservoir Update for Dynamic Scenes", mOptions.temporalUpdateForDynamicScene);

            mRecompile |= group.checkbox("Use Prev Frame Scene Data", mOptions.usePrevFrameSceneData);
        }


        dirty |= widget.checkbox("Additional Spatial resampling", mOptions.useAdditionalSpatialResampling);
        dirty |= widget.var("Additional Iterations", mOptions.additionalSpatialIterations);

        if (auto group = widget.group("Spatial resampling", true))
        {
            dirty |= group.checkbox("Use spatial resampling", mOptions.useSpatialResampling);

            dirty |= group.var("Iterations", mOptions.spatialIterations, 0u, 16u);
            group.tooltip("Number of spatial resampling iterations.");

            mReallocate |= group.var("Neighbor count", mOptions.spatialNeighborCount, 1u, 8u);
            group.tooltip("Number of neighbor samples to resample per pixel and iteration.");

            dirty |= group.var("Gather radius", mOptions.spatialGatherRadius, 0u, 40u);
            group.tooltip("Radius to gather samples from.");
        }

        if (auto group = widget.group("Debugging"))
        {
            mRecompile |= group.dropdown("Debug output", kDebugOutputList, reinterpret_cast<uint32_t&>(mOptions.debugOutput));
            mpPixelDebug->renderUI(group);

        }

        mRecompile |= mReallocate;
        dirty |= mRecompile;

        if (dirty) mResetTemporalReservoirs = true;

        return dirty;
    }

    void ReSTIRPathTracing::setOptions(const Options& options)
    {
        if (std::memcmp(&options, &mOptions, sizeof(Options)) != 0)
        {
            mOptions = options;
            mRecompile = true;
        }
    }

    void ReSTIRPathTracing::beginFrame(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles, bool needRecompile)
    {
        mRecompile |= needRecompile;

        mFrameDim = frameDim;

        prepareResources(pRenderContext, frameDim, screenTiles);

        mpPixelDebug->beginFrame(pRenderContext, mFrameDim);
    }

    void ReSTIRPathTracing::endFrame(RenderContext* pRenderContext)
    {
        mFrameIndex++;

        // Swap reservoirs.
        if (!mpScene->freeze)
        {
            std::swap(mpPrefixReservoirs, mpPrevPrefixReservoirs);
            std::swap(mpReservoirs, mpPrevReservoirs);
            std::swap(mpPrefixGBuffer, mpPrevPrefixGBuffer);
        }

        mpPixelDebug->endFrame(pRenderContext);
    }


    void ReSTIRPathTracing::updateReSTIRPT(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors, const ref<Texture>& pVBuffer)
    {
        FALCOR_PROFILE(pRenderContext, "ReSTIRPathTracing::updateReSTIRPT");

        if (mOptions.useAdditionalSpatialResampling)
            additionalSpatialResampling(pRenderContext, pVBuffer);

        temporalResampling(pRenderContext, pMotionVectors, pVBuffer);
        spatialResampling(pRenderContext, pVBuffer);

        // prepare temporal data
        if (!mpScene->freeze)
        {
            if (mpTemporalVBuffer)
                pRenderContext->copyResource(mpTemporalVBuffer.get(), pVBuffer.get());
            mPrevCameraU = mpScene->getCamera()->getData().cameraU;
            mPrevCameraV = mpScene->getCamera()->getData().cameraV;
            mPrevCameraW = mpScene->getCamera()->getData().cameraW;
            mPrevJitterX = mpScene->getCamera()->getData().jitterX;
            mPrevJitterY = mpScene->getCamera()->getData().jitterY;
        }
        return;
    }

    void ReSTIRPathTracing::createOrDestroyBuffer(ref<Buffer>& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition)
    {
        if (keepCondition && (mReallocate || !pBuffer || pBuffer->getElementCount() != requiredElementCount))
            pBuffer = mpDevice->createStructuredBuffer(mpReflectTypes->getRootVar()[reflectVarName], requiredElementCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
        if (!keepCondition) pBuffer = nullptr;
    }

    void ReSTIRPathTracing::createOrDestroyBufferWithCounter(ref<Buffer>& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition)
    {
        if (keepCondition && (mReallocate || !pBuffer || pBuffer->getElementCount() != requiredElementCount))
            pBuffer = mpDevice->createStructuredBuffer(mpReflectTypes->getRootVar()[reflectVarName], requiredElementCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, true);
        if (!keepCondition) pBuffer = nullptr;
    }

    void ReSTIRPathTracing::createOrDestroyBufferNoReallocate(ref<Buffer>& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition)
    {
        if (keepCondition && (!pBuffer || pBuffer->getElementCount() != requiredElementCount))
            pBuffer = mpDevice->createStructuredBuffer(mpReflectTypes->getRootVar()[reflectVarName], requiredElementCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
        if (!keepCondition) pBuffer = nullptr;
    }

    void ReSTIRPathTracing::createOrDestroyBufferWithCounterNoReallocate(ref<Buffer>& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition)
    {
        if (keepCondition && (!pBuffer || pBuffer->getElementCount() != requiredElementCount))
            pBuffer = mpDevice->createStructuredBuffer(mpReflectTypes->getRootVar()[reflectVarName], requiredElementCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, true);
        if (!keepCondition) pBuffer = nullptr;
    }

    void ReSTIRPathTracing::createOrDestroyRawBuffer(ref<Buffer>& pBuffer, size_t requiredSize, bool keepCondition)
    {
        if (keepCondition && (mReallocate || !pBuffer || pBuffer->getSize() != requiredSize))
            pBuffer = mpDevice->createBuffer(requiredSize, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr);
        if (!keepCondition) pBuffer = nullptr;
    }

    void ReSTIRPathTracing::prepareResources(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles)
    {
        // disable hybrid shift, temporal
        if (mReallocate && mpReservoirs) updatePrograms();
        // Create screen sized buffers.
        uint32_t tileCount = screenTiles.x * screenTiles.y;
        const uint32_t elementCount = tileCount * kScreenTileDim.x * kScreenTileDim.y;

        // getting correct struct sizes when initializing
        if (!mpReservoirs)
        {
            DefineList defines;
            defines.add(getDefines());
            TypeConformanceList typeConformances;
            // Scene-specific configuration.
            typeConformances.add(mpScene->getTypeConformances());
            ProgramDesc baseDesc;
            baseDesc.addShaderModules(mpScene->getShaderModules());
            baseDesc.addTypeConformances(typeConformances);
            baseDesc.setShaderModel(kShaderModel);
            createComputePass(mpReflectTypes, kReflectTypesFile, defines, baseDesc);
        }

        createOrDestroyBuffer(mpReservoirs, "pathReservoirs", elementCount);
        createOrDestroyBuffer(mpPrevReservoirs, "pathReservoirs", elementCount);
        createOrDestroyBuffer(mpScratchReservoirs, "pathReservoirs", elementCount);
        createOrDestroyBuffer(mpPrefixPathReservoirs, "pathReservoirs", elementCount);
        createOrDestroyBuffer(mpPrefixThroughputs, "prefixThroughputs", elementCount);

        createOrDestroyBuffer(mpPrevSuffixReservoirs, "pathReservoirs", elementCount, mOptions.subpathReuse);
        createOrDestroyBuffer(mpTempReservoirs, "pathReservoirs", elementCount, mpScene->freeze || mOptions.useAdditionalSpatialResampling);
        createOrDestroyBuffer(mpNeighborValidMaskBuffer, "neighborValidMask", elementCount);

        // for hybrid shift workload compaction
        int maxNeighborCount = std::max(std::max((int)mOptions.spatialNeighborCount, mOptions.subpathSetting.finalGatherSuffixCount), std::max(mOptions.subpathSetting.suffixSpatialNeighborCount, 1));
        const uint32_t talbotPathCount = elementCount * (mOptions.subpathSetting.useTalbotMISForGather ? mOptions.subpathSetting.finalGatherSuffixCount * (mOptions.subpathSetting.finalGatherSuffixCount + 1) : 0);
        const uint32_t pathCount = std::max(talbotPathCount, elementCount * 2 * maxNeighborCount);

        createOrDestroyRawBuffer(mpWorkload, pathCount * sizeof(uint32_t), mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact);
        createOrDestroyRawBuffer(mpWorkloadExtra, pathCount * sizeof(uint32_t), mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact && mOptions.subpathSetting.useTalbotMISForGather);

        createOrDestroyRawBuffer(mpCounter, sizeof(uint32_t), mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact);

        //4*1024*1024*1024 / 48 (rcReconnectionData size at compress reservoir and sd_optim), and round to nearest 10^7 (somehow using originla number causes crash)
        createOrDestroyBuffer(mpReconnectionDataBuffer, "reconnectionDataBuffer", std::min(80000000u, pathCount), (mOptions.shiftMapping == ReSTIRPT::ShiftMapping::Hybrid || mOptions.subpathReuse));
        createOrDestroyBuffer(mpRcBufferOffsets, "rcBufferOffsets", pathCount, (mOptions.shiftMapping == ReSTIRPT::ShiftMapping::Hybrid || mOptions.subpathReuse));

        const uint32_t maxVertexCount = elementCount * (mStaticParams.maxSurfaceBounces + 1);

        createOrDestroyBuffer(mpPrefixGBuffer, "prefixGBuffer", elementCount, mOptions.subpathReuse);
        createOrDestroyBuffer(mpPrevPrefixGBuffer, "prefixGBuffer", elementCount, mOptions.subpathReuse);
        createOrDestroyBuffer(mpFinalGatherSearchKeys, "prefixSearchKeys", elementCount, mOptions.subpathReuse);

        createOrDestroyBuffer(mpPrefixReservoirs, "prefixReservoirs", elementCount, mOptions.subpathReuse);
        createOrDestroyBuffer(mpPrevPrefixReservoirs, "prefixReservoirs", elementCount, mOptions.subpathReuse);

        createOrDestroyBuffer(mpScratchPrefixGBuffer, "prefixGBuffer", elementCount, mOptions.subpathReuse);

        int maxTemporalOrFinalGatherNeighborCount = std::max(mOptions.subpathSetting.finalGatherSuffixCount, 1);
        createOrDestroyBuffer(mpTemporalNeighborPixels, "temporalNeighborPixels", maxTemporalOrFinalGatherNeighborCount * elementCount,
            mOptions.subpathReuse);

        createOrDestroyBufferWithCounterNoReallocate(mpSearchPointBoundingBoxBuffer, "searchPointBoundingBoxBuffer", frameDim.x * frameDim.y, mOptions.subpathReuse);
        createOrDestroyBufferNoReallocate(mpPrefixL2LengthBuffer, "prefixL2LengthBuffer", frameDim.x * frameDim.y, mOptions.subpathReuse);

        if (!mOptions.subpathReuse) mpSearchASBuilder = nullptr;

        if (!mpTemporalVBuffer || mpTemporalVBuffer->getHeight() != frameDim.y || mpTemporalVBuffer->getWidth() != frameDim.x)
        {
            mpTemporalVBuffer = mpDevice->createTexture2D(frameDim.x, frameDim.y, mpScene->getHitInfo().getFormat(), 1, 1);
        }

        mReallocate = false;
    }

    void ReSTIRPathTracing::updatePrograms()
    {
         if (!mRecompile) return;

         DefineList commonDefines;

         commonDefines.add(getDefines());
         commonDefines.add(mpScene->getSceneDefines());
         commonDefines.add(mOwnerDefines);
         commonDefines.add("DEBUG_OUTPUT", std::to_string((uint32_t)mOptions.debugOutput));

         TypeConformanceList typeConformances;
         // Scene-specific configuration.
         typeConformances.add(mpScene->getTypeConformances());

         ProgramDesc baseDesc;
         baseDesc.addShaderModules(mpScene->getShaderModules());
         baseDesc.addTypeConformances(typeConformances);
         baseDesc.setShaderModel(kShaderModel);

         DefineList defines = commonDefines;
         defines.add("NEIGHBOR_OFFSET_COUNT", std::to_string(mpNeighborOffsets->getWidth()));

         createComputePass(mpReflectTypes, kReflectTypesFile, defines, baseDesc);
         createComputePass(mpPrefixResampling, kPrefixResampling, defines, baseDesc);
         createComputePass(mpTraceNewSuffixes, kTraceNewSuffixes, defines, baseDesc);
         createComputePass(mpTraceNewPrefixes, kTraceNewPrefixes, defines, baseDesc);
         createComputePass(mpPrefixNeighborSearch, kPrefixNeighborSearch, defines, baseDesc);
         createComputePass(mpSuffixSpatialResampling, kSuffixSpatialResamplingFile, defines, baseDesc, "spatial");
         createComputePass(mpSuffixTemporalResampling, kSuffixTemporalResamplingFile, defines, baseDesc, "temporal");
         createComputePass(mpSuffixResampling, kSuffixResamplingFile, defines, baseDesc, "gather");
         createComputePass(mpPrefixRetrace, kPrefixRetraceFile, defines, baseDesc);
         createComputePass(mpPrefixProduceRetraceWorkload, kPrefixProduceRetraceWorkload, defines, baseDesc);
         createComputePass(mpSuffixRetrace, kSuffixRetraceFile, defines, baseDesc);
         createComputePass(mpSuffixProduceRetraceWorkload, kSuffixProduceRetraceWorkload, defines, baseDesc);
         createComputePass(mpSuffixRetraceTalbot, kSuffixRetraceTalbotFile, defines, baseDesc);
         createComputePass(mpSuffixProduceRetraceTalbotWorkload, kSuffixProduceRetraceTalbotWorkload, defines, baseDesc);

         createComputePass(mpTemporalRetrace, kTemporalRetraceFile, defines, baseDesc);
         createComputePass(mpTemporalResampling, kTemporalResamplingFile, defines, baseDesc);
         createComputePass(mpProduceRetraceWorkload, kProduceRetraceWorkload, defines, baseDesc);
         createComputePass(mpSpatialRetrace, kSpatialRetraceFile, defines, baseDesc);
         createComputePass(mpSpatialResampling, kSpatialResamplingFile, defines, baseDesc);

         mRecompile = false;
         mResetTemporalReservoirs = true;
    }

    void ReSTIRPathTracing::suffixResamplingPass(
        RenderContext* pRenderContext,
        const ref<Texture>& pVBuffer,
        const ref<Texture>& pMotionVectors,
        const ref<Texture>& pVizBuffer,
        const ref<Texture>& pOutputColor,
        const ref<Texture>& pOutputSubColor
    )
    {
        FALCOR_PROFILE(pRenderContext, "SuffixResampling");

        bool hasTemporalReuse = mOptions.subpathSetting.suffixTemporalReuse;
        // if we have no temporal history, skip the first round (set suffixTemporalReuse in CB to false temporarily)
        mOptions.subpathSetting.suffixTemporalReuse = (mResetTemporalReservoirs) ? false : mOptions.subpathSetting.suffixTemporalReuse;

        ShaderVar presamplingVar = bindSuffixResamplingVars(pRenderContext, mpPrefixResampling, "gPrefixResampling", pVBuffer, pMotionVectors, true, true);
        presamplingVar["prevCameraU"] = mPrevCameraU;
        presamplingVar["prevCameraV"] = mPrevCameraV;
        presamplingVar["prevCameraW"] = mPrevCameraW;
        presamplingVar["prevJitterX"] = mPrevJitterX;
        presamplingVar["prevJitterY"] = mPrevJitterY;
        presamplingVar["prefixReservoirs"] = mpPrefixReservoirs;
        presamplingVar["prevPrefixReservoirs"] = mpPrevPrefixReservoirs;
        presamplingVar["rcBufferOffsets"] = mpRcBufferOffsets;
        presamplingVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;

        ShaderVar spatialVar = bindSuffixResamplingVars(pRenderContext, mpSuffixSpatialResampling, "gSuffixResampling", pVBuffer, pMotionVectors, true, false);
        spatialVar["outColor"] = pOutputColor;
        spatialVar["rcBufferOffsets"] = mpRcBufferOffsets;
        spatialVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        spatialVar["temporalNeighborPixels"] = mpTemporalNeighborPixels;

        ShaderVar temporalVar = bindSuffixResamplingVars(pRenderContext, mpSuffixTemporalResampling, "gSuffixResampling", pVBuffer, pMotionVectors, true, false);
        temporalVar["outColor"] = pOutputColor;
        temporalVar["rcBufferOffsets"] = mpRcBufferOffsets;
        temporalVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        temporalVar["temporalNeighborPixels"] = mpTemporalNeighborPixels;

        temporalVar["prevCameraU"] = mPrevCameraU;
        temporalVar["prevCameraV"] = mPrevCameraV;
        temporalVar["prevCameraW"] = mPrevCameraW;
        temporalVar["prevJitterX"] = mPrevJitterX;
        temporalVar["prevJitterY"] = mPrevJitterY;

        ShaderVar prefixVar = bindSuffixResamplingVars(pRenderContext, mpSuffixResampling, "gSuffixResampling", pVBuffer, pMotionVectors, true, true);
        prefixVar["outColor"] = pOutputColor;
        prefixVar["rcBufferOffsets"] = mpRcBufferOffsets;
        prefixVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        prefixVar["temporalNeighborPixels"] = mpTemporalNeighborPixels;

        ShaderVar workloadVar;
        if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact)
        {
            workloadVar = bindSuffixResamplingVars(
                pRenderContext, mpSuffixProduceRetraceWorkload, "gPathGenerator", pVBuffer, pMotionVectors, false, false
            );
            workloadVar["queue"]["counter"] = mpCounter;
            workloadVar["queue"]["workload"] = mpWorkload;
            workloadVar["temporalNeighborPixels"] = mpTemporalNeighborPixels;
        }

        ShaderVar retraceVar = bindSuffixResamplingVars(
            pRenderContext, mpSuffixRetrace, "gSuffixPathRetrace", pVBuffer, pMotionVectors, true, false
        );
        retraceVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        retraceVar["rcBufferOffsets"] = mpRcBufferOffsets;
        retraceVar["queue"]["counter"] = mpCounter;
        retraceVar["queue"]["workload"] = mpWorkload;
        retraceVar["temporalNeighborPixels"] = mpTemporalNeighborPixels;

        ShaderVar workloadVarTalbot;
        if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact && mOptions.subpathSetting.useTalbotMISForGather)
        {
            workloadVarTalbot = bindSuffixResamplingVars(
                pRenderContext, mpSuffixProduceRetraceTalbotWorkload, "gPathGenerator", pVBuffer, pMotionVectors, false, false
            );
            workloadVarTalbot["queue"]["counter"] = mpCounter;
            workloadVarTalbot["queue"]["workload"] = mpWorkload;
            workloadVarTalbot["queue"]["workloadExtra"] = mpWorkloadExtra;
            workloadVarTalbot["temporalNeighborPixels"] = mpTemporalNeighborPixels;
        }

        ShaderVar retraceVarTalbot = bindSuffixResamplingVars(
            pRenderContext, mpSuffixRetraceTalbot, "gSuffixPathRetrace", pVBuffer, pMotionVectors, true, false
        );
        retraceVarTalbot["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        retraceVarTalbot["rcBufferOffsets"] = mpRcBufferOffsets;
        retraceVarTalbot["queue"]["counter"] = mpCounter;
        retraceVarTalbot["queue"]["workload"] = mpWorkload;
        retraceVarTalbot["queue"]["workloadExtra"] = mpWorkloadExtra;
        retraceVarTalbot["temporalNeighborPixels"] = mpTemporalNeighborPixels;

        ShaderVar prefixWorkloadVar;
        if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact)
        {
            prefixWorkloadVar = bindPrefixResamplingVars(
                pRenderContext, mpPrefixProduceRetraceWorkload, "gPathGenerator", pVBuffer, pMotionVectors, false
            );
            prefixWorkloadVar["queue"]["counter"] = mpCounter;
            prefixWorkloadVar["queue"]["workload"] = mpWorkload;
        }

        ShaderVar prefixRetraceVar = bindPrefixResamplingVars(
            pRenderContext, mpPrefixRetrace, "gPrefixPathRetrace", pVBuffer, pMotionVectors, true
        );
        prefixRetraceVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        prefixRetraceVar["rcBufferOffsets"] = mpRcBufferOffsets;
        prefixRetraceVar["queue"]["counter"] = mpCounter;
        prefixRetraceVar["queue"]["workload"] = mpWorkload;
        prefixRetraceVar["prefixReservoirs"] = mpPrefixReservoirs;
        prefixRetraceVar["prevPrefixReservoirs"] = mpPrevPrefixReservoirs;
        prefixRetraceVar["prefixTotalLengthBuffer"] = mpPrefixL2LengthBuffer; // abuse the storage for this

        // try to re-bind the correct value for a term used to offset RNG
        int numRoundsForComputeRNG = mOptions.subpathSetting.suffixSpatialReuseRounds + 1 + (hasTemporalReuse ? 1 : 0);
        spatialVar["restirpt"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
        temporalVar["restirpt"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
        prefixVar["restirpt"]["suffixSpatialRounds"] = numRoundsForComputeRNG;

        if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact)
        {
            workloadVar["restirpt"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
            prefixWorkloadVar["restirpt"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
            if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact && mOptions.subpathSetting.useTalbotMISForGather)
                workloadVarTalbot["restirpt"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
        }
        retraceVar["restirpt"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
        prefixRetraceVar["restirpt"]["suffixSpatialRounds"] = numRoundsForComputeRNG;
        if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact && mOptions.subpathSetting.useTalbotMISForGather)
            retraceVarTalbot["restirpt"]["suffixSpatialRounds"] = numRoundsForComputeRNG;

        if (mOptions.subpathSetting.adaptivePrefixLength)
        {
            if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact)
            {
                FALCOR_PROFILE(pRenderContext, "ProducePrefixWorkload");

                if (mpCounter)
                    pRenderContext->clearUAV(mpCounter->getUAV().get(), uint4(0));

                prefixWorkloadVar["prevReservoirs"] = mpPrevSuffixReservoirs;
                const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
                mpPrefixProduceRetraceWorkload->execute(
                    pRenderContext, mPathTracerParams.screenTiles.x * tileSize, mPathTracerParams.screenTiles.y, 1
                );
            }

            {
                FALCOR_PROFILE(pRenderContext, "PrefixRetrace");
                prefixRetraceVar["prevReservoirs"] = mpPrevSuffixReservoirs;

                mpPrefixRetrace->execute(pRenderContext,
                    mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Naive ? mFrameDim.x : 2 * mFrameDim.x * mFrameDim.y,
                    mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Naive ? mFrameDim.y : 1, 1);
            }
        }

        {
            FALCOR_PROFILE(pRenderContext, "PrefixResampling");

            presamplingVar["reservoirs"] = mpReservoirs;
            presamplingVar["prevReservoirs"] = mpPrevSuffixReservoirs;
            presamplingVar["prefixSearchKeys"] = mpFinalGatherSearchKeys;
            presamplingVar["searchPointBoundingBoxBuffer"] = mpSearchPointBoundingBoxBuffer;
            presamplingVar["prefixTotalLengthBuffer"] = mpPrefixL2LengthBuffer;
            presamplingVar["screenSpacePixelSpreadAngle"] = mpScene->getCamera()->computeScreenSpacePixelSpreadAngle(mFrameDim.y);

            mpPrefixResampling->execute(
                pRenderContext, mFrameDim.x, mFrameDim.y, 1
            );
        }

        if (!mpSearchASBuilder)
        {
            mpSearchASBuilder = BoundingBoxAccelerationStructureBuilder::Create(mpSearchPointBoundingBoxBuffer, mpDevice);
        }

        if (!mResetTemporalReservoirs)
        {
            FALCOR_PROFILE(pRenderContext, "BuildSearchAS");
            uint numSearchPoints = mFrameDim.x * mFrameDim.y;
            mpSearchASBuilder->BuildAS(pRenderContext, numSearchPoints, 1);
        }

        // trace an additional path
        {
            FALCOR_PROFILE(pRenderContext, "TraceNewSuffixes");

            // Bind global resources.
            auto var = mpTraceNewSuffixes->getRootVar();
            mpScene->setRaytracingShaderData(pRenderContext, var);
            mpPixelDebug->prepareProgram(mpTraceNewSuffixes->getProgram(), var);
            mpPixelStats->prepareProgram(mpTraceNewSuffixes->getProgram(), mpTraceNewSuffixes->getRootVar());

            // Bind the path tracer.
            var["gPathTracer"] = mpPathTracerBlock;
            var["gScheduler"]["prefixGbuffer"] = mpPrefixGBuffer;
            var["gScheduler"]["pathReservoirs"] = mpReservoirs;
            // Full screen dispatch.
            mpTraceNewSuffixes->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
        }

        int numLevels = 1;

        for (int iter = 0; iter < numLevels; iter++)
        {
            int numRounds = mOptions.subpathSetting.suffixSpatialReuseRounds;
            // the actual rounds used
            numRounds = mOptions.subpathSetting.suffixTemporalReuse ? numRounds + 1 : numRounds;

            for (int i = 0; i < numRounds; i++)
            {
                bool isCurrentPassTemporal = mOptions.subpathSetting.suffixTemporalReuse && i == 0;
                ref<Buffer>& pPrevSuffixReservoirs = (isCurrentPassTemporal || !mpScene->freeze) ? mpPrevSuffixReservoirs : mpTempReservoirs;

                if (!isCurrentPassTemporal)
                {
                    std::swap(mpReservoirs, pPrevSuffixReservoirs);
                }

                if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact)
                {
                    FALCOR_PROFILE(pRenderContext, isCurrentPassTemporal ? "TemporalSuffixProduceRetraceWorkload" : "SpatialSuffixProduceRetraceWorkload");

                    if (mpCounter)
                        pRenderContext->clearUAV(mpCounter->getUAV().get(), uint4(0));

                    workloadVar["reservoirs"] = mpReservoirs;
                    workloadVar["prevReservoirs"] = pPrevSuffixReservoirs;
                    workloadVar["suffixReuseRoundId"] = i;
                    workloadVar["curPrefixLength"] = numLevels - iter;

                    const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
                    mpSuffixProduceRetraceWorkload->execute(
                        pRenderContext, mPathTracerParams.screenTiles.x * tileSize, mPathTracerParams.screenTiles.y, 1
                    );
                }

                {
                    FALCOR_PROFILE(pRenderContext, isCurrentPassTemporal ? "TemporalSuffixRetrace" : "SpatialSuffixRetrace");

                    retraceVar["reservoirs"] = mpReservoirs;
                    retraceVar["prevReservoirs"] = pPrevSuffixReservoirs;
                    retraceVar["suffixReuseRoundId"] = i;

                    if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Naive)
                        mpSuffixRetrace->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
                    else
                        mpSuffixRetrace->execute(pRenderContext, 2 * (isCurrentPassTemporal ? 1 : mOptions.subpathSetting.suffixSpatialNeighborCount) * mFrameDim.x * mFrameDim.y, 1, 1);
                }


                {
                    FALCOR_PROFILE(pRenderContext, isCurrentPassTemporal ? "TemporalSuffixResampling" : "SpatialSuffixResampling");

                    ShaderVar& tempVar = isCurrentPassTemporal ? temporalVar : spatialVar;
                    ref<ComputePass>& tempPass = isCurrentPassTemporal ? mpSuffixTemporalResampling : mpSuffixSpatialResampling;

                    tempVar["reservoirs"] = mpReservoirs;
                    tempVar["prevReservoirs"] = pPrevSuffixReservoirs;
                    tempVar["suffixReuseRoundId"] = i;
                    tempVar["curPrefixLength"] = numLevels - iter;
                    tempVar["vbuffer"] = pVBuffer;

                    tempPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
                }
            }

            mOptions.subpathSetting.suffixTemporalReuse = hasTemporalReuse;


            //  generate multiple suffixes
            ref<Buffer>& pPrevSuffixReservoirs = !mpScene->freeze ? mpPrevSuffixReservoirs : mpTempReservoirs;
            std::swap(mpReservoirs, pPrevSuffixReservoirs);

            for (int integrationPrefixId = 0; integrationPrefixId < mOptions.subpathSetting.numIntegrationPrefixes; integrationPrefixId++)
            {
                bool hasCanonicalSuffix =
                    mOptions.subpathSetting.generateCanonicalSuffixForEachPrefix ? true : integrationPrefixId == 0;

                // we borrow the prefix of integrationPrefixId 0 from before
                {
                    // trace new prefixes
                    FALCOR_PROFILE(pRenderContext, "TraceNewPrefixes");

                    // Bind global resources.
                    auto var = mpTraceNewPrefixes->getRootVar();
                    mpScene->setRaytracingShaderData(pRenderContext, var);
                    mpPixelDebug->prepareProgram(mpTraceNewPrefixes->getProgram(), var);
                    mpPixelStats->prepareProgram(mpTraceNewPrefixes->getProgram(), var);
                    // Bind the path tracer.
                    var["gPathTracer"] = mpPathTracerBlock;
                    var["gScheduler"]["integrationPrefixId"] = integrationPrefixId;
                    var["gScheduler"]["shouldGenerateSuffix"] = hasCanonicalSuffix;
                    // Full screen dispatch.
                    mpTraceNewPrefixes->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
                }

                // stream prefixes
                {
                    FALCOR_PROFILE(pRenderContext, "FinalGather");

                    {
                        FALCOR_PROFILE(pRenderContext, "FinalGatherNeighborSearch");

                        auto rootVar = mpPrefixNeighborSearch->getRootVar();
                        mpScene->bindShaderData(rootVar["gScene"]);
                        mpPixelDebug->prepareProgram(mpPrefixNeighborSearch->getProgram(), rootVar);
                        auto var = rootVar["CB"]["gPrefixNeighborSearch"];

                        var["neighborOffsets"] = mpNeighborOffsets;
                        var["motionVectors"] = pMotionVectors;
                        var["params"].setBlob(mPathTracerParams);
                        setShaderData(var["restirpt"]);
                        var["prefixGBuffer"] = mpScratchPrefixGBuffer;
                        var["prevPrefixGBuffer"] = mpPrefixGBuffer;
                        var["temporalNeighborPixels"] = mpTemporalNeighborPixels;
                        var["integrationPrefixId"] = integrationPrefixId;
                        var["prefixSearchKeys"] = mpFinalGatherSearchKeys;
                        var["hasSearchPointAS"] = !mResetTemporalReservoirs;
                        var["searchPointBoundingBoxBuffer"] = mpSearchPointBoundingBoxBuffer;

                        if (mpSearchASBuilder && !mResetTemporalReservoirs)
                            mpSearchASBuilder->SetRaytracingShaderData(var, "gSearchPointAS", 1u);

                        mpPrefixNeighborSearch->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
                    }

                    ref<ComputePass> pFinalGatherRetraceProduceWorkload = mOptions.subpathSetting.useTalbotMISForGather ?
                        mpSuffixProduceRetraceTalbotWorkload : mpSuffixProduceRetraceWorkload;

                    if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact)
                    {
                        FALCOR_PROFILE(pRenderContext, "FinalGatherProduceRetraceWorkload");

                        if (mpCounter)
                            pRenderContext->clearUAV(mpCounter->getUAV().get(), uint4(0));

                        ShaderVar& var_ = mOptions.subpathSetting.useTalbotMISForGather ? workloadVarTalbot : workloadVar;

                        var_["prevReservoirs"] = pPrevSuffixReservoirs;
                        var_["suffixReuseRoundId"] = -1;
                        var_["integrationPrefixId"] = integrationPrefixId;

                        const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
                        pFinalGatherRetraceProduceWorkload->execute(
                            pRenderContext, mPathTracerParams.screenTiles.x * tileSize, mPathTracerParams.screenTiles.y, 1
                        );
                    }

                    ref<ComputePass> pSuffixRetrace = mOptions.subpathSetting.useTalbotMISForGather ?
                        mpSuffixRetraceTalbot: mpSuffixRetrace;

                    {
                        FALCOR_PROFILE(pRenderContext, "FinalGatherSuffixRetrace");

                        ShaderVar& var_ = mOptions.subpathSetting.useTalbotMISForGather ? retraceVarTalbot : retraceVar;

                        var_["prevReservoirs"] = pPrevSuffixReservoirs;
                        var_["suffixReuseRoundId"] = -1;
                        var_["integrationPrefixId"] = integrationPrefixId;

                        int multiplier = mOptions.subpathSetting.useTalbotMISForGather ? mOptions.subpathSetting.finalGatherSuffixCount + 1 : 2;

                        if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Naive)
                            pSuffixRetrace->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
                        else
                            pSuffixRetrace->execute(pRenderContext, multiplier * mOptions.subpathSetting.finalGatherSuffixCount * mFrameDim.x * mFrameDim.y, 1, 1);
                    }

                    {
                        FALCOR_PROFILE(pRenderContext, "FinalGatherIntegration");

                        prefixVar["reservoirs"] = mpReservoirs;
                        prefixVar["prevReservoirs"] = pPrevSuffixReservoirs;
                        prefixVar["suffixReuseRoundId"] = -1;
                        prefixVar["prefixReservoirs"] = mpPrefixReservoirs;
                        prefixVar["curPrefixLength"] = numLevels - iter;
                        prefixVar["integrationPrefixId"] = integrationPrefixId;
                        prefixVar["hasCanonicalSuffix"] = hasCanonicalSuffix;

                        mpSuffixResampling->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
                    }
                }
            }
        }

        if (!mOptions.useTemporalResampling || mOptions.subpathSetting.directlyOutputColor)
            mResetTemporalReservoirs = false;
    }

    ShaderVar ReSTIRPathTracing::bindSpatialResamplingVars(RenderContext* pRenderContext, ref<ComputePass> pPass, std::string cbName, const ref<Texture>& pVBuffer, bool bindPathTracer, bool isAdditionalSpatial)
    {
        auto rootVar = pPass->getRootVar();
        mpScene->bindShaderData(rootVar["gScene"]);
        mpScene->setRaytracingShaderData(pRenderContext, rootVar);
        mpPixelDebug->prepareProgram(pPass->getProgram(), rootVar);
        mpPixelStats->prepareProgram(pPass->getProgram(), rootVar);

        auto var = rootVar["CB"][cbName];
        var["neighborOffsets"] = mpNeighborOffsets;
        var["gNumSpatialRounds"] = mOptions.spatialIterations;
        var["gNeighborCount"] = mOptions.spatialNeighborCount;
        var["gGatherRadius"] = (float)mOptions.spatialGatherRadius;
        var["gAdditionalSpatial"] = isAdditionalSpatial;
        var["vbuffer"] = pVBuffer;

        var["params"].setBlob(mPathTracerParams);
        setShaderData(var["restirpt"]);

        if (bindPathTracer)
        {
            rootVar["gPathTracer"] = mpPathTracerBlock;
        }

        return var;
    }


    void ReSTIRPathTracing::spatialResampling(RenderContext* pRenderContext, const ref<Texture>& pVBuffer)
    {
        FALCOR_PROFILE(pRenderContext, "spatialResampling");

        if (!mOptions.useSpatialResampling) return;

        ShaderVar var = bindSpatialResamplingVars(pRenderContext, mpSpatialResampling, "gSpatialResampling", pVBuffer, true, false);
        var["neighborValidMask"] = mpNeighborValidMaskBuffer;
        var["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        var["rcBufferOffsets"] = mpRcBufferOffsets;

        ShaderVar retraceVar = bindSpatialResamplingVars(pRenderContext, mpSpatialRetrace, "gSpatialPathRetrace", pVBuffer, true, false);
        retraceVar["neighborValidMask"] = mpNeighborValidMaskBuffer;
        retraceVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        retraceVar["queue"]["counter"] = mpCounter;
        retraceVar["queue"]["workload"] = mpWorkload;
        retraceVar["rcBufferOffsets"] = mpRcBufferOffsets;

        ShaderVar workLoadVar;
        if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact)
        {
            workLoadVar = bindSpatialResamplingVars(pRenderContext, mpProduceRetraceWorkload, "gPathGenerator", pVBuffer, false, false);
            workLoadVar["queue"]["counter"] = mpCounter;
            workLoadVar["queue"]["workload"] = mpWorkload;
            workLoadVar["neighborValidMask"] = mpNeighborValidMaskBuffer;
            workLoadVar["isSpatialPass"] = true;
        }

        ref<Buffer>& pSwapReservoir = mpScene->freeze ? mpTempReservoirs : mpPrevReservoirs;

        for (uint32_t iteration = 0; iteration < mOptions.spatialIterations; ++iteration)
        {
            std::swap(mpReservoirs, pSwapReservoir);

            if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact)
            {
                FALCOR_PROFILE(pRenderContext, "produceRetraceWorkload");

                if (mpCounter)
                    pRenderContext->clearUAV(mpCounter->getUAV().get(), uint4(0));

                workLoadVar["prevReservoirs"] = pSwapReservoir;
                workLoadVar["gSpatialRoundId"] = iteration;
                const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
                mpProduceRetraceWorkload->execute(pRenderContext, mPathTracerParams.screenTiles.x * tileSize, mPathTracerParams.screenTiles.y, 1);
            }

            {
                FALCOR_PROFILE(pRenderContext, "spatialRetrace");
                retraceVar["prevReservoirs"] = pSwapReservoir;
                retraceVar["gSpatialRoundId"] = iteration;
                mpSpatialRetrace->execute(pRenderContext,
                    mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Naive ? mFrameDim.x : 2 * mOptions.spatialNeighborCount * mFrameDim.x * mFrameDim.y,
                    mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Naive ? mFrameDim.y : 1 , 1);
            }

            {
                FALCOR_PROFILE(pRenderContext, "spatialResampling");
                var["reservoirs"] = mpReservoirs;
                var["prevReservoirs"] = pSwapReservoir;

                var["gSpatialRoundId"] = iteration;
                mpSpatialResampling->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
            }
        }
    }

    void ReSTIRPathTracing::additionalSpatialResampling(RenderContext* pRenderContext, const ref<Texture>& pVBuffer)
    {
        FALCOR_PROFILE(pRenderContext, "additionalSpatialResampling");

        if (!mOptions.useAdditionalSpatialResampling) return;

        ShaderVar var = bindSpatialResamplingVars(pRenderContext, mpSpatialResampling, "gSpatialResampling", pVBuffer, true, true);
        var["neighborValidMask"] = mpNeighborValidMaskBuffer;
        var["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        var["rcBufferOffsets"] = mpRcBufferOffsets;

        ShaderVar retraceVar = bindSpatialResamplingVars(pRenderContext, mpSpatialRetrace, "gSpatialPathRetrace", pVBuffer, true, true);
        retraceVar["neighborValidMask"] = mpNeighborValidMaskBuffer;
        retraceVar["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
        retraceVar["rcBufferOffsets"] = mpRcBufferOffsets;
        retraceVar["queue"]["counter"] = mpCounter;
        retraceVar["queue"]["workload"] = mpWorkload;

        ShaderVar workLoadVar;
        if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact)
        {
            workLoadVar = bindSpatialResamplingVars(pRenderContext, mpProduceRetraceWorkload, "gPathGenerator", pVBuffer, false, true);
            workLoadVar["queue"]["counter"] = mpCounter;
            workLoadVar["queue"]["workload"] = mpWorkload;
            workLoadVar["neighborValidMask"] = mpNeighborValidMaskBuffer;
            workLoadVar["isSpatialPass"] = true;
        }

        ref<Buffer>& pSwapReservoir = mpTempReservoirs;

        for (uint32_t iteration = 0; iteration < mOptions.additionalSpatialIterations; ++iteration)
        {
            std::swap(mpReservoirs, pSwapReservoir);

            if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact)
            {
                FALCOR_PROFILE(pRenderContext, "produceRetraceWorkload");

                if (mpCounter)
                    pRenderContext->clearUAV(mpCounter->getUAV().get(), uint4(0));

                workLoadVar["prevReservoirs"] = pSwapReservoir;
                workLoadVar["gSpatialRoundId"] = iteration;
                const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
                mpProduceRetraceWorkload->execute(pRenderContext, mPathTracerParams.screenTiles.x * tileSize, mPathTracerParams.screenTiles.y, 1);
            }

            {
                FALCOR_PROFILE(pRenderContext, "spatialRetrace");
                retraceVar["prevReservoirs"] = pSwapReservoir;
                retraceVar["gSpatialRoundId"] = iteration;
                mpSpatialRetrace->execute(pRenderContext,
                    mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Naive ? mFrameDim.x : 2 * mOptions.spatialNeighborCount * mFrameDim.x * mFrameDim.y,
                    mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Naive ? mFrameDim.y : 1, 1);
            }

            {
                FALCOR_PROFILE(pRenderContext, "spatialResampling");
                var["reservoirs"] = mpReservoirs;
                var["prevReservoirs"] = pSwapReservoir;

                var["gSpatialRoundId"] = iteration;
                mpSpatialResampling->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
            }
        }
    }

    ShaderVar ReSTIRPathTracing::bindSuffixResamplingVars(RenderContext* pRenderContext,
        ref<ComputePass> pPass, std::string cbName, const ref<Texture>& pVBuffer, const ref<Texture>& pMotionVectors, bool bindPathTracer, bool bindVBuffer)
    {
        auto rootVar = pPass->getRootVar();
        mpScene->bindShaderData(rootVar["gScene"]);

        mpScene->setRaytracingShaderData(pRenderContext, rootVar);
        mpPixelDebug->prepareProgram(pPass->getProgram(), rootVar);
        mpPixelStats->prepareProgram(pPass->getProgram(), rootVar);

        auto var = rootVar["CB"][cbName];

        var["neighborOffsets"] = mpNeighborOffsets;

        var["motionVectors"] = pMotionVectors;

        var["params"].setBlob(mPathTracerParams);
        setShaderData(var["restirpt"]);

        // Bind the path tracer.
        if (bindPathTracer)
        {
            rootVar["gPathTracer"] = mpPathTracerBlock;
        }

        var["reservoirs"] = mpReservoirs;
        var["prevReservoirs"] = !mpScene->freeze ? mpPrevSuffixReservoirs : mpTempReservoirs; // doesn't matter, this will be binded differently for different passes

        var["prefixGBuffer"] = mpPrefixGBuffer;
        var["prevPrefixGBuffer"] = mpPrevPrefixGBuffer;

        var["neighborValidMask"] = mpNeighborValidMaskBuffer;

        if (bindVBuffer)
        {
            var["vbuffer"] = pVBuffer;
            var["temporalVbuffer"] = mpTemporalVBuffer;
        }
        return var;
    }


    ShaderVar ReSTIRPathTracing::bindSuffixResamplingOneVars(RenderContext* pRenderContext,
        ref<ComputePass> pPass, std::string cbName, const ref<Texture>& pMotionVectors, bool bindPathTracer)
    {
        auto rootVar = pPass->getRootVar();
        mpScene->bindShaderData(rootVar["gScene"]);

        mpScene->setRaytracingShaderData(pRenderContext, rootVar);
        mpPixelDebug->prepareProgram(pPass->getProgram(), rootVar);
        mpPixelStats->prepareProgram(pPass->getProgram(), rootVar);

        auto var = rootVar["CB"][cbName];

        var["motionVectors"] = pMotionVectors;

        var["params"].setBlob(mPathTracerParams);
        setShaderData(var["restirpt"]);

        // Bind the path tracer.
        if (bindPathTracer)
        {
            rootVar["gPathTracer"] = mpPathTracerBlock;
        }

        var["reservoirs"] = mpReservoirs;
        var["prevReservoirs"] = mpPrevSuffixReservoirs;

        var["prefixGBuffer"] = mpPrefixGBuffer;

        return var;
    }


    ShaderVar ReSTIRPathTracing::bindPrefixResamplingVars(RenderContext* pRenderContext, ref<ComputePass> pPass, std::string cbName, const ref<Texture>& pVBuffer, const ref<Texture>& pMotionVectors, bool bindPathTracer)
    {
        auto rootVar = pPass->getRootVar();
        mpScene->bindShaderData(rootVar["gScene"]);

        mpScene->setRaytracingShaderData(pRenderContext, rootVar);
        mpPixelDebug->prepareProgram(pPass->getProgram(), rootVar);
        mpPixelStats->prepareProgram(pPass->getProgram(), rootVar);

        auto var = rootVar["CB"][cbName];

        var["vbuffer"] = pVBuffer;
        var["temporalVbuffer"] = mpTemporalVBuffer;

        var["motionVectors"] = pMotionVectors;

        var["prevCameraU"] = mPrevCameraU;
        var["prevCameraV"] = mPrevCameraV;
        var["prevCameraW"] = mPrevCameraW;
        var["prevJitterX"] = mPrevJitterX;
        var["prevJitterY"] = mPrevJitterY;

        var["params"].setBlob(mPathTracerParams);
        setShaderData(var["restirpt"]);

        var["neighborValidMask"] = mpNeighborValidMaskBuffer;

        // Bind the path tracer.
        if (bindPathTracer)
        {
            rootVar["gPathTracer"] = mpPathTracerBlock;
        }

        return var;
    }

    ShaderVar ReSTIRPathTracing::bindTemporalResamplingVars(RenderContext* pRenderContext, ref<ComputePass> pPass, std::string cbName, const ref<Texture>& pVBuffer, const ref<Texture>& pMotionVectors, bool bindPathTracer)
    {
        auto rootVar = pPass->getRootVar();
        mpScene->bindShaderData(rootVar["gScene"]);

        mpScene->setRaytracingShaderData(pRenderContext, rootVar);
        mpPixelDebug->prepareProgram(pPass->getProgram(), rootVar);
        mpPixelStats->prepareProgram(pPass->getProgram(), rootVar);

        auto var = rootVar["CB"][cbName];

        var["gNumSpatialRounds"] = mOptions.spatialIterations;

        var["vbuffer"] = pVBuffer;
        var["temporalVbuffer"] = mpTemporalVBuffer;

        var["motionVectors"] = pMotionVectors;
        var["gEnableTemporalReprojection"] = true;
        var["gNoResamplingForTemporalReuse"] = false;
        var["gTemporalHistoryLength"] = mOptions.maxHistoryLength;

        var["prevCameraU"] = mPrevCameraU;
        var["prevCameraV"] = mPrevCameraV;
        var["prevCameraW"] = mPrevCameraW;
        var["prevJitterX"] = mPrevJitterX;
        var["prevJitterY"] = mPrevJitterY;

        var["params"].setBlob(mPathTracerParams);
        setShaderData(var["restirpt"]);

        // Bind the path tracer.
        if (bindPathTracer)
        {
            rootVar["gPathTracer"] = mpPathTracerBlock;
        }

        var["reservoirs"] = mpReservoirs;
        var["prevReservoirs"] = mpPrevReservoirs;
        return var;
    }


    void ReSTIRPathTracing::temporalResampling(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors, const ref<Texture>& pVBuffer)
    {
        FALCOR_PROFILE(pRenderContext, "temporalResampling");

        if (mResetTemporalReservoirs)
        {
            mResetTemporalReservoirs = false;
            return;
        }

        if (!mOptions.useTemporalResampling) return;


        if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Compact)
        {
            FALCOR_PROFILE(pRenderContext, "ProduceRetraceWorkload");

            if (mpCounter)
                pRenderContext->clearUAV(mpCounter->getUAV().get(), uint4(0));

            ShaderVar var = bindTemporalResamplingVars(
                pRenderContext, mpProduceRetraceWorkload, "gPathGenerator", pVBuffer, pMotionVectors, false
            );
            var["queue"]["counter"] = mpCounter;
            var["queue"]["workload"] = mpWorkload;
            var["neighborValidMask"] = mpNeighborValidMaskBuffer;
            var["isSpatialPass"] = false;
            var["gAdditionalSpatial"] = false;

            const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
            mpProduceRetraceWorkload->execute(
                pRenderContext, mPathTracerParams.screenTiles.x * tileSize, mPathTracerParams.screenTiles.y, 1
            );
        }

        {
            FALCOR_PROFILE(pRenderContext, "temporalRetrace");

            ShaderVar var = bindTemporalResamplingVars(
                pRenderContext, mpTemporalRetrace, "gTemporalPathRetrace", pVBuffer, pMotionVectors, true
            );
            var["neighborValidMask"] = mpNeighborValidMaskBuffer;
            var["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
            var["rcBufferOffsets"] = mpRcBufferOffsets;
            var["queue"]["counter"] = mpCounter;
            var["queue"]["workload"] = mpWorkload;

            if (mOptions.retraceScheduleType == ReSTIRPT::RetraceScheduleType::Naive)
                mpTemporalRetrace->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
            else
                mpTemporalRetrace->execute(pRenderContext, 2 * mFrameDim.x * mFrameDim.y, 1, 1);
        }

        {
            FALCOR_PROFILE(pRenderContext, "temporalResampling");

            ShaderVar var = bindTemporalResamplingVars(pRenderContext, mpTemporalResampling, "gTemporalResampling", pVBuffer, pMotionVectors, true);
            var["neighborValidMask"] = mpNeighborValidMaskBuffer;
            var["reconnectionDataBuffer"] = mpReconnectionDataBuffer;
            var["rcBufferOffsets"] = mpRcBufferOffsets;

            mpTemporalResampling->execute(pRenderContext, mFrameDim.x, mFrameDim.y, 1);
        }
    }

    ref<Texture> ReSTIRPathTracing::createNeighborOffsetTexture(uint32_t sampleCount)
    {
        std::unique_ptr<int8_t[]> offsets(new int8_t[sampleCount * 2]);
        const int R = 254;
        const float phi2 = 1.f / 1.3247179572447f;
        float u = 0.5f;
        float v = 0.5f;
        for (uint32_t index = 0; index < sampleCount * 2;)
        {
            u += phi2;
            v += phi2 * phi2;
            if (u >= 1.f) u -= 1.f;
            if (v >= 1.f) v -= 1.f;

            float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
            if (rSq > 0.25f) continue;

            offsets[index++] = int8_t((u - 0.5f) * R);
            offsets[index++] = int8_t((v - 0.5f) * R);
        }

        return mpDevice->createTexture1D(sampleCount, ResourceFormat::RG8Snorm, 1, 1, offsets.get());
    }

}
