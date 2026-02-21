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
#include "Utils/Sampling/AliasTable.h"
#include "Utils/Debug/PixelDebug.h"
#include "Rendering/Utils/PixelStats.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Scene/Scene.h"
#include "Scene/Lights/LightCollection.h"
#include "Scene/Lights/Light.h"
#include "ReSTIRPT.slang"
#include "Params.slang"
#include <cmath>
#include <memory>
#include <random>
#include <tuple>
#include <vector>
#include "BoundingBoxAccelerationStructureBuilder.h"

namespace Falcor
{
    /** Implementation of ReSTIR Path Tracing (ReSTIR PT).

        based on
        "Generalized Resampled Importance Sampling: Foundations of ReSTIR"
        by Lin et al. from 2022.
    */
    class FALCOR_API ReSTIRPathTracing
    {
    public:

        /** Enumeration of available debug outputs.
            Note: Keep in sync with definition in Params.slang
        */
        enum class DebugOutput
        {
            Disabled,
            Position,
            Depth,
            Normal,
            FaceNormal,
            DiffuseWeight,
            SpecularWeight,
            SpecularRoughness,
            PackedNormal,
            PackedDepth,
            InitialWeight,
            TemporalReuse,
            SpatialReuse,
            FinalSampleDir,
            FinalSampleDistance,
            FinalSampleLi,
        };

        /** Configuration options.
        */
        struct Options
        {
            // Common Options for ReSTIR DI and GI.

            // Temporal resampling options.
            bool useTemporalResampling = true;          ///< Enable temporal resampling.
            bool temporalUpdateForDynamicScene = false;
            uint32_t maxHistoryLength = 32;             ///< Maximum temporal history length.

            // TODO: MIS option

            // Spatial resampling options.

            bool useAdditionalSpatialResampling = false;
            uint32_t additionalSpatialIterations = 1;             ///< Number of spatial resampling iterations.

            bool useSpatialResampling = true;           ///< Enable spatial resampling.
            uint32_t spatialIterations = 1;             ///< Number of spatial resampling iterations.
            uint32_t spatialNeighborCount = 2;          ///< Number of neighbor samples to resample per pixel and iteration.
            uint32_t spatialGatherRadius = 32;          ///< Radius to gather samples from.

            // TODO: MIS option

            // General options.
            DebugOutput debugOutput = DebugOutput::Disabled;

            // Options for ReSTIR PT.
            ReSTIRPT::ShiftMappingSettings shiftMappingSettings;

            uint32_t reservoirCountPerPixel = 1;                ///< Number of reservoirs per pixel.

            // static params
            ReSTIRPT::ShiftMapping shiftMapping = ReSTIRPT::ShiftMapping::Hybrid;

            bool useReservoirCompression = true;

            // subpath reuse
            bool subpathReuse = false;

            uint32_t minimumPrefixLength = 1;

            ReSTIRPT::SubpathReuseSettings subpathSetting;

            // subpath reuse general settings

            bool visualizeFireflies = false;

            bool usePrevFrameSceneData = false;

            ReSTIRPT::RetraceScheduleType retraceScheduleType = ReSTIRPT::RetraceScheduleType::Compact;

            // Note: Empty constructor needed for clang due to the use of the nested struct constructor in the parent constructor.
            Options() {}

            template<typename Archive>
            void serialize(Archive& ar)
            {
                ar("useTemporalResampling", useTemporalResampling);
                ar("maxHistoryLength", maxHistoryLength);
                ar("useSpatialResampling", useSpatialResampling);
                ar("spatialIterations", spatialIterations);
                ar("spatialNeighborCount", spatialNeighborCount);
                ar("spatialGatherRadius", spatialGatherRadius);
                ar("subpathReuse", subpathReuse);
                ar("subpathSetting", subpathSetting);
                ar("shiftMappingSettings", shiftMappingSettings);
            }
        };

        // static params shared with internal path tracer
        struct SharedStaticParams
        {
            uint32_t    samplesPerPixel;                        ///< Number of samples (paths) per pixel, unless a sample density map is used.
            uint32_t    maxSurfaceBounces;                      ///< Max number of surface bounces (diffuse + specular + transmission), up to kMaxPathLenth. This will be initialized at startup.
            bool        useNEE;                              ///< Use next-event estimation (NEE). This enables shadow ray(s) from each path vertex.
        };

        ReSTIRPathTracing(const ref<Scene>& pScene, const DefineList& ownerDefines, const Options& options, const std::shared_ptr<PixelStats>& pPixelStats);

        /** Get a list of shader defines for using the ReSTIR sampler.
            \return Returns a list of defines.
        */
        DefineList getDefines() const;

        /** Bind the ReSTIR sampler to a given shader var.
            \param[in] var The shader variable to set the data into.
        */
        void setShaderData(const ShaderVar& var) const;

        void setReservoirData(const ShaderVar& var) const;

        /** Render the GUI.
            \return True if options were changed, false otherwise.
        */
        bool renderUI(Gui::Widgets& widget);

        /** Returns the current configuration.
        */
        Options& getOptions()  { return mOptions; }

        /** Set the configuration.
        */
        void setOptions(const Options& options);

        /** Begin a frame.
            Must be called once at the beginning of each frame.
            \param[in] pRenderContext Render context.
            \param[in] frameDim Current frame dimension.
        */
        void beginFrame(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles, bool needRecompile);

        /** End a frame.
            Must be called one at the end of each frame.
            \param[in] pRenderContext Render context.
        */
        void endFrame(RenderContext* pRenderContext);


        /** Update the ReSTIR sampler.
            This runs the ReSTIR PT algorithm.
            Must be called once between beginFrame() and endFrame().
            \param[in] pRenderContext Render context.
            \param[in] pMotionVectors Motion vectors for temporal reprojection.
        */
        void updateReSTIRPT(RenderContext* pRenderContext, const ref<Texture>& vBuffer, const ref<Texture>& pMotionVectors);

        /** Get the debug output texture.
            \return Returns the debug output texture.
        */
        const ref<Texture>& getDebugOutputTexture() const { return mpDebugOutputTexture; }

        /** Get the pixel debug component.
            \return Returns the pixel debug component.
        */
        const std::unique_ptr<PixelDebug>& getPixelDebug() const { return mpPixelDebug; }

        void setPathTracerParams(int useFixedSeed, uint fixedSeed,
            float lodBias, float specularRoughnessThreshold, uint2 frameDim, uint2 screenTiles, uint frameCount, uint seed, int samplesPerPixel, int DIMode);

        void setOwnerDefines(DefineList defines);

        void setSharedStaticParams(uint32_t samplesPerPixel, uint32_t maxSurfaceBounces, bool useNEE);

        void createPathTracerBlock();

        ref<ParameterBlock> getPathTracerBlock();

        void updatePrograms();

        void suffixResamplingPass(
            RenderContext* pRenderContext,
            const ref<Texture>& pVBuffer,
            const ref<Texture>& pMotionVectors,
            const ref<Texture>& pVizBuffer,
            const ref<Texture>& pOutputColor,
            const ref<Texture>& pOutputSubColor
        );

        bool needResetTemporalHistory() { return mResetTemporalReservoirs;  }

    private:

        void prepareResources(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles);

        void temporalResampling(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors, const ref<Texture>& pVBuffer);
        void spatialResampling(RenderContext* pRenderContext, const ref<Texture>& pVBuffer);
        void additionalSpatialResampling(RenderContext* pRenderContext, const ref<Texture>& pVBuffer);

        void createOrDestroyBuffer(ref<Buffer>& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition=true);
        void createOrDestroyBufferWithCounter(ref<Buffer>& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition = true);
        void createOrDestroyBufferNoReallocate(ref<Buffer>& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition = true);
        void createOrDestroyBufferWithCounterNoReallocate(ref<Buffer>& pBuffer, std::string reflectVarName, int requiredElementCount, bool keepCondition = true);

        void createOrDestroyRawBuffer(ref<Buffer>& pBuffer, size_t requiredSize, bool keepCondition=true);

        Falcor::ShaderVar bindSpatialResamplingVars(RenderContext* pRenderContext, ref<ComputePass> pPass, std::string cbName, const ref<Texture>& pVBuffer, bool bindPathTracer, bool isAdditionalSpatial);
        Falcor::ShaderVar bindTemporalResamplingVars(RenderContext* pRenderContext, ref<ComputePass> pPass, std::string cbName, const ref<Texture>& pVBuffer, const ref<Texture>& pMotionVectors, bool bindPathTracer);
        Falcor::ShaderVar bindSuffixResamplingVars(RenderContext* pRenderContext, ref<ComputePass> pPass, std::string cbName, const ref<Texture>& pVBuffer, const ref<Texture>& pMotionVectors, bool bindPathTracer, bool bindVBuffer);
        Falcor::ShaderVar bindPrefixResamplingVars(RenderContext* pRenderContext, ref<ComputePass> pPass, std::string cbName, const ref<Texture>& pVBuffer, const ref<Texture>& pMotionVectors, bool bindPathTracer);
        Falcor::ShaderVar bindSuffixResamplingOneVars(RenderContext* pRenderContext, ref<ComputePass> pPass, std::string cbName, const ref<Texture>& pMotionVectors, bool bindPathTracer);
        /** Create a 1D texture with random offsets within a unit circle around (0,0).
            The texture is RG8Snorm for compactness and has no mip maps.
            \param[in] sampleCount Number of samples in the offset texture.
        */
        ref<Texture> createNeighborOffsetTexture(uint32_t sampleCount);

        void createComputePass(ref<ComputePass>& pPass, std::string shaderFile, DefineList defines, ProgramDesc desc, std::string entryFunction="");

        ref<Scene> mpScene;                           ///< Scene.
        ref<Device>                         mpDevice; ///< GPU device.

        Options mOptions;                                   ///< Configuration options.
        SharedStaticParams mStaticParams;

        ReSTIRPathTracerParams                mPathTracerParams;                    ///< agrees with that in InlinePathTracer

        DefineList mOwnerDefines;                   ///< Share defines with inline path tracer

        std::mt19937 mRng;                                  ///< Random generator.


        std::shared_ptr<PixelStats>           mpPixelStats;               ///< Utility class for collecting pixel stats. (shared with host renderpass)
        std::unique_ptr<PixelDebug> mpPixelDebug;                 ///< Pixel debug component.

        uint2 mFrameDim = uint2(0);                         ///< Current frame dimensions.
        uint32_t mFrameIndex = 0;                           ///< Current frame index.

        ref<ComputePass> mpReflectTypes;              ///< Pass for reflecting types.

        // ReSTIR PT passes.

        ref<ComputePass> mpPrefixProduceRetraceWorkload;
        ref<ComputePass> mpPrefixRetrace;
        ref<ComputePass> mpSuffixSpatialResampling;
        ref<ComputePass> mpSuffixTemporalResampling;
        ref<ComputePass> mpSuffixResampling;
        ref<ComputePass> mpTemporalResampling;        ///< Pass for temporal resampling.
        ref<ComputePass> mpTemporalRetrace;           ///< Pass for temporal resampling.
        ref<ComputePass> mpSpatialResampling;         ///< Pass for spatial resampling.
        ref<ComputePass> mpSpatialRetrace;            ///< Pass for spatial resampling.
        ref<ComputePass> mpProduceRetraceWorkload;
        ref<ComputePass> mpSuffixRetrace;
        ref<ComputePass> mpSuffixProduceRetraceWorkload;
        ref<ComputePass> mpSuffixRetraceTalbot;
        ref<ComputePass> mpSuffixProduceRetraceTalbotWorkload;
        ref<ComputePass> mpPrefixResampling;
        ref<ComputePass> mpTraceNewSuffixes;
        ref<ComputePass> mpTraceNewPrefixes;
        ref<ComputePass> mpPrefixNeighborSearch;

        ref<ParameterBlock>       mpPathTracerBlock;          ///< Parameter block for the path tracer.

        ref<Buffer> mpScratchVerticesBuffer;          ///< Buffer containing the temporary sample (one candidate in initial RIS)'s path vertices

        ref<Buffer> mpPrefixGBuffer;          ///< Buffer containing the current sample's path vertices
        ref<Buffer> mpPrevPrefixGBuffer;

        ref<Buffer> mpPrefixReservoirs;
        ref<Buffer> mpPrevPrefixReservoirs;

        ref<Buffer> mpScratchPrefixGBuffer;

        ref<Buffer> mpScratchReservoirs;              ///< Buffer containing the temporary reservoirs. // can also hold firefly reservoirs/prev suffix reservoirs
        ref<Buffer> mpReservoirs;                     ///< Buffer containing the current reservoirs.
        ref<Buffer> mpPrefixPathReservoirs;
        ref<Buffer> mpPrefixThroughputs;
        ref<Buffer> mpPrevReservoirs;                 ///< Buffer containing the previous reservoirs.
        ref<Buffer> mpPrevSuffixReservoirs;           ///< Buffer containing previous suffix reservoirs.
        ref<Buffer> mpTemporalNeighborPixels;

        // TODO: fix this for directlyOutputColor
        ref<Buffer> mpTempReservoirs;                 ///< can hold both initial sampling results and firefly path reserovirs
        ref<Buffer> mpReconnectionDataBuffer;          ///< Buffer containing the reconnection data for retrace result.
        ref<Buffer> mpRcBufferOffsets;
        ref<Buffer> mpNeighborValidMaskBuffer;

        ref<Buffer> mpFinalGatherSearchKeys;

        ref<Buffer>               mpWorkload;             ///< Paths starting from primary hits on general materials (all types).
        ref<Buffer>               mpWorkloadExtra;             ///< Paths starting from primary hits on general materials (all types).
        ref<Buffer>               mpCounter;                 ///< Atomic counters (32-bit).

        ref<Texture> mpDebugOutputTexture;            ///< Debug output texture.
        ref<Texture> mpNeighborOffsets;               ///< 1D texture containing neighbor offsets within a unit circle.

        ref<Buffer> mpSearchPointBoundingBoxBuffer;
        ref<Buffer> mpPrefixL2LengthBuffer;
        ref<BoundingBoxAccelerationStructureBuilder> mpSearchASBuilder;

        // temporal data.
        float3 mPrevCameraU;
        float3 mPrevCameraV;
        float3 mPrevCameraW;
        float mPrevJitterX;
        float mPrevJitterY;
        ref<Texture> mpTemporalVBuffer;

public:
        bool mRecompile = true;                             ///< Recompile programs on next frame if set to true.
        bool mReallocate = true;                            ///< Reallocate the reservoirs since sizes change
        bool mResetTemporalReservoirs = true;               ///< Reset temporal reservoir buffer on next frame if set to true.

    };
}
