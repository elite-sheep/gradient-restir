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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Rendering/Lights/EmissiveLightSampler.h"
#include "Rendering/Lights/EmissiveLightSamplerType.slangh"
#include "Utils/Debug/PixelDebug.h"

#include "DiffIntegrator.h"
#include "DiffIntegratorType.slangh"
#include "Params.slang"

#include <random>

using namespace Falcor;

class GPathTracer : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(
        GPathTracer,
        "GPathTracer",
        "Gradient domain Path Tracer"
    );

    static ref<GPathTracer> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<GPathTracer>(pDevice, props);
    }

    GPathTracer(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void setScene(RenderContext* pRenderContext,
                          const ref<Scene>& pScene) override;
    virtual void execute(RenderContext* pRenderContext,
                         const RenderData& renderData) override;
    virtual void renderUI(RenderContext* pRenderContext,
                          Gui::Widgets& widget) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    static void registerBindings(pybind11::module& m);

private:
    // Static configuration. Change any of them will cause recompilation.
    struct StaticParams
    {
        // Path tracing parameters
        uint32_t samplesPerPixel = 1;
        uint32_t maxBounces = 2;
        bool     useAlphaTest = true;
        bool     useNEE = true;      // Next event estimation
        EmissiveLightSamplerType emissiveLightSamplerType = EmissiveLightSamplerType::Power;
        uint32_t sampleGenerator = SAMPLE_GENERATOR_TINY_UNIFORM;
        uint32_t diffIntegrator = DIFF_INTEGRATOR_MULTI_POINT_RESTIR;

        // ReSTIR parameters
        uint32_t numInitialSamples = 1;
        bool useTemporalReuse = true;
        bool useSpatialReuse = true;
        uint32_t numSpatialNeighbours = 2;
        uint shiftMappingType = 0;

        // Poisson reconstruction parameters
        uint32_t poissonSolverIterations = 48;
        float    poissonSolverAlpha = 0.2f;

        DefineList getDefines(const GPathTracer& owner) const;
    };

private:
    bool beginFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void endFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareTracePass(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareBasePaths(RenderContext* pRenderContext, const RenderData& renderData);
    bool prepareLighting(RenderContext* pRenderContext);
    bool prepareEmissiveLighting(RenderContext* pRenderContext);
    void prepareSampleGenerator(RenderContext* pRenderContext);
    void prepareDiffIntegrator(RenderContext* pRenderContext);
    void prepareReconstructionPass(RenderContext* pRenderContext, const RenderData& renderData);
    void preparePrimaryEmissionPass(RenderContext* pRenderContext, const RenderData& renderData);
    void preparePrepareBasePathsPass(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);
    bool renderRenderingUI(Gui::Widgets& widget);
    void setFrameDim(const uint2 frameDim);
    void sampleInitialSamples(RenderContext* pRenderContext, const RenderData& renderData);
    void temporalReuse(RenderContext* pRenderContext, const RenderData& renderData);
    void spatialReuse(RenderContext* pRenderContext, const RenderData& renderData);
    void tracePass(RenderContext* pRenderContext, const RenderData& renderData);
    void computePrimaryEmission(RenderContext* pRenderContext, const RenderData& renderData);
    void updatePrograms();

    // ReSTIR
    void createNeighborOffsetTexture(uint32_t sampleCount);

    // Poisson solver
    void reconstruct(RenderContext* pRenderContext, const RenderData& renderData);

    // Helper function for GPU memory
    void create2DTexture(ref<Texture>& texture, uint2 dim, ResourceFormat format);
    void createBuffer(ref<Buffer>& pBuffer, std::string reflectVarName, int requiredElementCount);
    void createParameterBlock(ref<ParameterBlock>& pParameterBlock, std::string reflectVarName);

private:
    ref<Scene> mpScene;
    ref<DiffIntegrator> mpDXDiffIntegrator;
    ref<DiffIntegrator> mpDYDiffIntegrator;
    std::unique_ptr<EmissiveLightSampler> mpEmissiveLigthSampler;
    GPathTracerParams mParams;
    StaticParams mStaticParams;

    // sampler
    uint mpMetaSeed = 2027;
    std::mt19937 mpGen;
    std::uniform_int_distribution<unsigned int> mpDistr;
    ref<SampleGenerator> mpSampleGenerator;

    // Types
    ref<ComputePass> mpReflectPass;

    // Trace pass
    ref<ComputePass> mpPrimaryEmissionPass;
    ref<ComputePass> mpPrepareBasePathsPass;
    ref<ComputePass> mpPrimalTemporalLookForNeighboursPass;
    ref<ComputePass> mpPrimalTemporalReusePass;
    ref<ComputePass> mpPrimalSpatialReusePass;
    ref<ComputePass> mpEvaluatePass;
    ref<Buffer> mpPrimalReservoirs;
    ref<Buffer> mpPrimalSampledTemporalReservoirs;
    ref<Buffer> mpPrimalPaths;
    ref<Texture> mpPrimalTemporalFracPixelLocs;
    ref<Texture> mpPrimalCanFindTemporalNeighbors;

    // Gradient passes
    ref<ComputePass> mpDXTemporalReusePass;
    ref<ComputePass> mpDXTemporalNegReusePass;
    ref<ComputePass> mpDXTemporalLookForNeighboursPass;
    ref<ComputePass> mpDXSpatialReusePass;
    ref<ComputePass> mpDXSpatialNegReusePass;
    ref<ComputePass> mpDXSpatialLookForNeighboursPass;
    ref<ComputePass> mpDYTemporalReusePass;
    ref<ComputePass> mpDYTemporalNegReusePass;
    ref<ComputePass> mpDYTemporalLookForNeighboursPass;
    ref<ComputePass> mpDYSpatialReusePass;
    ref<ComputePass> mpDYSpatialNegReusePass;
    ref<ComputePass> mpDYSpatialLookForNeighboursPass;

    // Poisson solver
    ref<Texture> mpReconstructedLast;
    ref<ComputePass> mpPoissonSolverPass;

    bool mNeedsToRecompile;
    bool mVarsChanged = true;

    // GPathTracer
    ref<ParameterBlock> mpGPathTracerBlock;
    CameraData mpPrevCameraData;

    // ReSTIR related
    ref<ParameterBlock> mpPrimalReSTIR;
    ref<ParameterBlock> mpDXIntegrator;
    ref<ParameterBlock> mpDYIntegrator;
    ref<Texture> mpNeighborOffsets;
    ref<Texture> mpPrimaryEmission;
    ref<Texture> mpPrimaryAlbedo;
    ref<Texture> mpPrimaryNormalAndDepth;
    ref<Buffer> mpVBuffer;
    ref<Buffer> mpPrevVBuffer;
    ref<Buffer> mpBasePaths;
    ref<Buffer> mpBasePathsRad;
    ref<Buffer> mpDXShiftedPathsRad;
    ref<Buffer> mpDYShiftedPathsRad;
    ref<Texture> mpDXSpatialNeighbours;
    ref<Texture> mpDYSpatialNeighbours;

    std::unique_ptr<PixelDebug> mpPixelDebug;

    uint pathsPerPixel = 1;
};
