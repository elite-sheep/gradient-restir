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
#include "GPathTracer.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/LightBVHSampler.h"

namespace
{
const char kPrimalTemporalLookForNeighboursShaderFile[] = "RenderPasses/GPathTracer/PrimalTemporalLookForNeighbour.cs.slang";
const char kPrimalTemporalReusePassShaderFile[] = "RenderPasses/GPathTracer/PrimalTemporalReuse.cs.slang";
const char kPrimalSpatialReusePassShaderFile[] = "RenderPasses/GPathTracer/PrimalSpatialReuse.cs.slang";

const char kMultiPointTemporalReuseShaderFile[] = "RenderPasses/GPathTracer/MultiPointTemporalReuse.cs.slang";
const char kMultiPointTemporalNegReuseShaderFile[] = "RenderPasses/GPathTracer/MultiPointTemporalNegReuse.cs.slang";
const char kMultiTemporalLookForNeighboursShaderFile[] = "RenderPasses/GPathTracer/MultiPointTemporalLookForNeighbour.cs.slang";
const char kMultiSpatialReuseShaderFile[] = "RenderPasses/GPathTracer/MultiPointSpatialReuse.cs.slang";
const char kMultiSpatialNegReuseShaderFile[] = "RenderPasses/GPathTracer/MultiPointSpatialNegReuse.cs.slang";
const char kMultiSpatialLookForNeighboursShaderFile[] = "RenderPasses/GPathTracer/MultiPointSpatialLookForNeighbour.cs.slang";

const char kPrimaryEmissionShaderFile[] = "RenderPasses/GPathTracer/PrimaryEmission.cs.slang";
const char kPrepareBasePathsShaderFile[] = "RenderPasses/GPathTracer/PrepareBasePaths.cs.slang";
const char kEvaluatePassShaderFile[]    = "RenderPasses/GPathTracer/EvaluatePass.cs.slang";
const char kReconstructPassShaderFile[] = "RenderPasses/GPathTracer/PoissonSolver.cs.slang";
const char kReflectTypesShaderFile[]    = "RenderPasses/GPathTracer/ReflectTypes.cs.slang";

// Render pass inputs and outputs.
const std::string kInputDepth = "depth";
const std::string kInputVBuffer = "vbuffer";
const std::string kInputVBufferCenter = "vbufferCenter";
const std::string kInputMotionVectors = "mvec";
const std::string kInputViewDir = "viewW";
const std::string kInputSubPixelUV = "subPixelUV";
const std::string kInputSampleCount = "sampleCount";
const std::string kOutputPrimal = "primal";
const std::string kOutputDX = "DX";
const std::string kOutputDY = "DY";
const std::string kOutputColor = "color";
const std::string kOutputAlbedo = "albedo";
const std::string kDXSpatialReuse = "DXSpatialReuse";
const std::string kDYSpatialReuse = "DYSpatialReuse";

const uint32_t kMaxPayloadSizeBytes = 72u;
const uint32_t kMaxRecursionDepth = 4u;

const ChannelList kInputChannels =
{
    { kInputDepth,          "gDepth",           "Depth buffer (NDC)",               true /* optional */, ResourceFormat::R32Float    },
    { kInputVBuffer,        "gVBuffer",         "Visibility buffer in packed format" },
    { kInputVBufferCenter,  "gVBufferCenter",   "Visibility buffer in packed format" },
    { kInputMotionVectors,  "gMotionVectors",   "Motion vector buffer (float format)", true /* optional */, ResourceFormat::RG32Float },
    { kInputViewDir,        "gViewW",           "World-space view direction (xyz float format)", true },
    { kInputSubPixelUV,     "gSubPixelUV",      "Sub-pixel UV (xyz float format)", true /* optional */, ResourceFormat::RG32Float },
    { kInputSampleCount,    "gSampleCount",     "Sample count buffer (integer format)", true /* optional */, ResourceFormat::R8Uint },
};

const ChannelList kOutputChannels = {
    { kOutputPrimal, "gOutputPrimal", "Primal output color", false, ResourceFormat::RGBA32Float},
    { kOutputDX,     "gOutputDX", "DX output color", false, ResourceFormat::RGBA32Float},
    { kOutputDY,     "gOutputDY", "DY output color", false, ResourceFormat::RGBA32Float},
    { kOutputColor,  "gOutputColor", "Final output color", false, ResourceFormat::RGBA32Float},
    { kDXSpatialReuse,  "gDXSpatialReuse", "Final output color", false, ResourceFormat::RGBA32Float},
    { kDYSpatialReuse,  "gDYSpatialReuse", "Final output color", false, ResourceFormat::RGBA32Float}
};

const uint32_t kNeighborOffsetCount = 8192;
const std::string kSamplesPerPixel = "samplesPerPixel";
const std::string kMaxBounces = "maxBounces";
const std::string kMetaSeed = "metaSeed";
const std::string kUseNEE = "useNEE";
const std::string kNumInitialSamples = "numInitialSamples";
const std::string kUseTemporalReuse = "useTemporalReuse";
const std::string kUseSpatialReuse = "useSpatialReuse";
const std::string kNumSpatialNeighbours = "numSpatialNeighbours";
const std::string kJacobianThreshold = "jacobianThreshold";
const std::string kSpatialNeighboursSearchRadius = "spatialNeighboursSearchRadius";
const std::string kShiftMappingType = "shiftMappingType";
const std::string kShouldRejectInnerShift = "shouldRejectInnerShift";
const std::string kShouldRejectOuterShift = "shouldRejectOuterShift";
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, GPathTracer>();
    ScriptBindings::registerBinding(GPathTracer::registerBindings);
}

void GPathTracer::registerBindings(pybind11::module& m)
{
    pybind11::class_<GPathTracer, RenderPass, ref<GPathTracer>> pass(m, "GPathTracer");
}

GPathTracer::GPathTracer(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice), mNeedsToRecompile(true)
{
    logInfo("GPathTracer:: Create a new pass.");
    for (const auto& [key, value] : props)
    {
        if (key == kSamplesPerPixel)
            mStaticParams.samplesPerPixel = value;
        else if (key == kMaxBounces)
            mStaticParams.maxBounces = value;
        else if (key == kUseNEE)
            mStaticParams.useNEE = value;
        else if (key == kNumInitialSamples)
            mStaticParams.numInitialSamples = value;
        else if (key == kUseTemporalReuse)
            mStaticParams.useTemporalReuse = value;
        else if (key == kUseSpatialReuse)
            mStaticParams.useSpatialReuse = value;
        else if (key == kNumSpatialNeighbours)
            mStaticParams.numSpatialNeighbours = value;
        else if (key == kJacobianThreshold)
            mParams.jacobianThreshold = value;
        else if (key == kSpatialNeighboursSearchRadius)
            mParams.spatialGatherRadius = value;
        else if (key == kMetaSeed)
            mpMetaSeed = value;
        else if (key == kShiftMappingType)
            mStaticParams.shiftMappingType = value;
        else if (key == kShouldRejectInnerShift)
            mParams.shouldRejectInnerShift = value;
        else if (key == kShouldRejectOuterShift)
            mParams.shouldRejectOuterShift = value;
        else
            logWarning("Unknown property '{}' in GPathTracer properties.", key);
    }

    // create spatial neighbor offsets
    createNeighborOffsetTexture(kNeighborOffsetCount);

    // Create random number generator
    mpGen = std::mt19937(mpMetaSeed);
    mpDistr = std::uniform_int_distribution<uint>();
    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);
    mpPixelDebug->enable();
}

Properties GPathTracer::getProperties() const
{
    Properties props;

    props[kSamplesPerPixel] = mStaticParams.samplesPerPixel;
    props[kMaxBounces] = mStaticParams.maxBounces;
    props[kUseNEE] = mStaticParams.useNEE;
    props[kNumInitialSamples] = mStaticParams.numInitialSamples;

    return props;
}

RenderPassReflection GPathTracer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void GPathTracer::setScene(RenderContext* pRenderContext,
                           const ref<Scene>& pScene)
{
    logInfo("GPathTracer::setScene(): set a new scene");
    mpScene = pScene;

    mParams.frameCount = 0;
    mParams.frameDim = {};

    mpPrimalTemporalLookForNeighboursPass = nullptr;
    mpPrimalTemporalReusePass = nullptr;
    mpPrimalSpatialReusePass = nullptr;
    mpPrimaryEmissionPass = nullptr;
    mpPrepareBasePathsPass = nullptr;
    mpPoissonSolverPass = nullptr;
    mpEmissiveLigthSampler = nullptr;
    mpSampleGenerator = nullptr;
    mpDXDiffIntegrator = nullptr;
    mpDYDiffIntegrator = nullptr;

    mpDXTemporalReusePass = nullptr;
    mpDXTemporalLookForNeighboursPass = nullptr;
    mpDXSpatialReusePass = nullptr;
    mpDXSpatialNegReusePass = nullptr;
    mpDXSpatialLookForNeighboursPass = nullptr;
    mpDYTemporalReusePass = nullptr;
    mpDYTemporalNegReusePass = nullptr;
    mpDYTemporalLookForNeighboursPass = nullptr;
    mpDYSpatialReusePass = nullptr;
    mpDYSpatialNegReusePass = nullptr;
    mpDYSpatialLookForNeighboursPass = nullptr;

    // Need to recompile the program when we reset the scene.
    if (mpScene)
    {
        mpScene->getCamera()->dumpProperties();
    }
}

void GPathTracer::execute(RenderContext* pRenderContext,
                          const RenderData& renderData)
{
    if (!beginFrame(pRenderContext, renderData))
        return;


    // Check if there is a scene
    if (mpScene == nullptr)
        return;

    // Prepare sample generator and lighting
    prepareSampleGenerator(pRenderContext);
    prepareDiffIntegrator(pRenderContext);

    bool lightingChanged = prepareLighting(pRenderContext);
    mNeedsToRecompile |= lightingChanged;

    updatePrograms();

    prepareResources(pRenderContext, renderData);
    prepareTracePass(pRenderContext, renderData);
    preparePrimaryEmissionPass(pRenderContext, renderData);
    preparePrepareBasePathsPass(pRenderContext, renderData);
    computePrimaryEmission(pRenderContext, renderData);
    prepareBasePaths(pRenderContext, renderData);
    tracePass(pRenderContext, renderData);

    // prepareReconstructionPass(pRenderContext, renderData);
    // reconstruct(pRenderContext, renderData);
    endFrame(pRenderContext, renderData);
}

void GPathTracer::renderUI(RenderContext* pRenderContext,
                           Gui::Widgets& widget)
{
    // TODO: Implement this.
    bool valueChanged = false;

    valueChanged |= renderRenderingUI(widget);
    if (auto group = widget.group("Debugging"))
    {
        mpPixelDebug->renderUI(group);
    }

    mNeedsToRecompile |= valueChanged;
}

void GPathTracer::preparePrimaryEmissionPass(RenderContext* pRenderContext,
                                            const RenderData& renderData)
{
    auto primaryEmissionPassVar = mpPrimaryEmissionPass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(primaryEmissionPassVar);
    }
    primaryEmissionPassVar["gGPathTracer"] = mpGPathTracerBlock;
    primaryEmissionPassVar["primaryEmission"] = mpPrimaryEmission;
    primaryEmissionPassVar["albedo"] = mpPrimaryAlbedo;
    primaryEmissionPassVar["normalAndDepth"] = mpPrimaryNormalAndDepth;
    mpScene->setRaytracingShaderData(pRenderContext, primaryEmissionPassVar);
    mpPixelDebug->prepareProgram(mpPrimaryEmissionPass->getProgram(), primaryEmissionPassVar);
}

void GPathTracer::preparePrepareBasePathsPass(RenderContext* pRenderContext,
                                            const RenderData& renderData)
{
    auto prepareBasePathsPassVar = mpPrepareBasePathsPass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(prepareBasePathsPassVar);
    }
    prepareBasePathsPassVar["gGPathTracer"] = mpGPathTracerBlock;
    prepareBasePathsPassVar["basePaths"] = mpBasePaths;
    prepareBasePathsPassVar["primaryHitInfo"] = mpVBuffer;
    prepareBasePathsPassVar["basePathsRadiance"] = mpBasePathsRad;
    prepareBasePathsPassVar["dxShiftedPathsRadiance"] = mpDXShiftedPathsRad;
    prepareBasePathsPassVar["dyShiftedPathsRadiance"] = mpDYShiftedPathsRad;
    mpScene->setRaytracingShaderData(pRenderContext, prepareBasePathsPassVar);
    mpPixelDebug->prepareProgram(mpPrepareBasePathsPass->getProgram(), prepareBasePathsPassVar);
}

void GPathTracer::prepareBasePaths(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "prepareBasePaths");
    mpPrepareBasePathsPass->execute(pRenderContext,
                                   { mParams.frameDim.x, mParams.frameDim.y, 1 });
}

void GPathTracer::computePrimaryEmission(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "computePrimaryEmission");
    mpPrimaryEmissionPass->execute(pRenderContext,
                                   { mParams.frameDim.x, mParams.frameDim.y, 1 });
}

void GPathTracer::sampleInitialSamples(RenderContext* pRenderContext, const RenderData& renderData)
{
}

void GPathTracer::temporalReuse(RenderContext* pRenderContext, const RenderData& renderData)
{
    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::primalTemporalLookForNeighbours()");
        mpPrimalTemporalLookForNeighboursPass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }

    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::primalTemporalReuse()");
        mpPrimalTemporalReusePass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }

    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::DXTemporalLookForNeighbours()");
        mpDXTemporalLookForNeighboursPass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }

    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::DXSpatialLookForNeighbours()");
        mpDXSpatialLookForNeighboursPass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }


    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::DXTemporalReuse()");
        mpDXTemporalReusePass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }

    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::DXTemporalNegReuse()");
        mpDXTemporalNegReusePass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }

    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::DYTemporalLookForNeighbours()");
        mpDYTemporalLookForNeighboursPass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }

    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::DYSpatialLookForNeighbours()");
        mpDYSpatialLookForNeighboursPass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }

    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::DYTemporalReuse()");
        mpDYTemporalReusePass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }

    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::DYTemporalNegReuse()");
        mpDYTemporalNegReusePass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }

}

void GPathTracer::spatialReuse(RenderContext* pRenderContext, const RenderData& renderData)
{
    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::spatialReuse()");
        mpPrimalSpatialReusePass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }

    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::DXSpatialReuse()");
        mpDXSpatialReusePass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }

    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::DXSpatialNegReuse()");
        mpDXSpatialNegReusePass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }

    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::DYSpatialReuse()");
        mpDYSpatialReusePass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }

    {
        FALCOR_PROFILE(pRenderContext, "GPathTracer::DYSpatialNegReuse()");
        mpDYSpatialNegReusePass->execute(pRenderContext,
                            { mParams.frameDim.x, mParams.frameDim.y, 1 });
    }
}

void GPathTracer::tracePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    this->sampleInitialSamples(pRenderContext, renderData);
    this->temporalReuse(pRenderContext, renderData);
    this->spatialReuse(pRenderContext, renderData);
    mpEvaluatePass->execute(pRenderContext,
                           { mParams.frameDim.x, mParams.frameDim.y, 1 });
}

void GPathTracer::reconstruct(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "Poisson Reconstruction");
    for (uint i = 0; i < mStaticParams.poissonSolverIterations; ++i)
    {
        mpPoissonSolverPass->execute(pRenderContext,
                                    { mParams.frameDim.x, mParams.frameDim.y, 1 });
        pRenderContext->copyResource(mpReconstructedLast.get(), renderData.getTexture(kOutputColor).get());
    }
}

bool GPathTracer::onMouseEvent(const MouseEvent& mouseEvent)
{
    bool dirty = mpPixelDebug->onMouseEvent(mouseEvent);
    return dirty;
}

bool GPathTracer::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Reset output buffers
    for (const auto& channel : kOutputChannels)
    {
        auto pTexture = renderData.getTexture(channel.name);
        pRenderContext->clearUAV(pTexture->getUAV().get(), float4(0.0f));
    }

    const auto& pOutputColor = renderData.getTexture(kOutputPrimal);
    FALCOR_ASSERT(pOutputColor);

    setFrameDim(uint2(pOutputColor->getWidth(), pOutputColor->getHeight()));

    // Validate all I/O sizes match the expected size
    bool resolutionMismatch = false;
    auto validateChannels = [&](const auto& channels)
    {
        for (const auto& channel : channels)
        {
            auto pTexture = renderData.getTexture(channel.name);
            if (pTexture && (pTexture->getWidth() != mParams.frameDim.x || pTexture->getHeight() != mParams.frameDim.y))
                resolutionMismatch = true;
        }
    };
    validateChannels(kInputChannels);
    validateChannels(kOutputChannels);
    if (resolutionMismatch)
    {
        logError("WARDiffPathTracer I/O sizes don't match. The pass will be disabled.");
    }

    mpPixelDebug->beginFrame(pRenderContext, mParams.frameDim);
    float3 sceneExtent = mpScene->getSceneBounds().extent();
    mParams.sceneRadius = std::min(sceneExtent.x, std::min(sceneExtent.y, sceneExtent.z));

    if (mParams.frameCount == 0)
    {
        mpPrevCameraData = mpScene->getCamera()->getData();
    }

    pathsPerPixel = 2;

    return true;
}

void GPathTracer::endFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mpScene)
    {
        mpPrevCameraData = mpScene->getCamera()->getData();
    }
    pRenderContext->copyResource(mpPrevVBuffer.get(), mpVBuffer.get());
    mpPixelDebug->endFrame(pRenderContext);

    mVarsChanged = false;
    mParams.frameCount++;
    mParams.seed = mpDistr(mpGen);
}

void GPathTracer::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpReconstructedLast || mVarsChanged)
    {
        create2DTexture(mpReconstructedLast, mParams.frameDim, ResourceFormat::RGBA32Float);
    }
    pRenderContext->clearUAV(mpReconstructedLast->getUAV().get(), float4(0.0f));

    if (!mpPrimaryEmission || mVarsChanged)
    {
        create2DTexture(mpPrimaryEmission, mParams.frameDim, ResourceFormat::RGBA32Float);
    }

    if (!mpPrimaryAlbedo || mVarsChanged)
    {
        create2DTexture(mpPrimaryAlbedo, mParams.frameDim, ResourceFormat::RGBA32Float);
    }

    if (!mpPrimaryNormalAndDepth || mVarsChanged)
    {
        create2DTexture(mpPrimaryNormalAndDepth, mParams.frameDim, ResourceFormat::RGBA32Float);
    }

    if (!mpPrimalTemporalFracPixelLocs || mVarsChanged)
    {
        create2DTexture(mpPrimalTemporalFracPixelLocs, mParams.frameDim, ResourceFormat::RG32Float);
    }

    if (!mpPrimalCanFindTemporalNeighbors || mVarsChanged)
    {
        create2DTexture(mpPrimalCanFindTemporalNeighbors, mParams.frameDim, ResourceFormat::R32Uint);
    }

    if (!mpDXSpatialNeighbours || mVarsChanged)
    {
        create2DTexture(mpDXSpatialNeighbours, mParams.frameDim, ResourceFormat::RG32Uint);
    }

    if (!mpDYSpatialNeighbours || mVarsChanged)
    {
        create2DTexture(mpDYSpatialNeighbours, mParams.frameDim, ResourceFormat::RG32Uint);
    }

    if (!mpVBuffer || mVarsChanged)
    {
        createBuffer(mpVBuffer, "vBuffer", mParams.frameDim.x * mParams.frameDim.y);
    }

    if (!mpPrevVBuffer || mVarsChanged)
    {
        createBuffer(mpPrevVBuffer, "vBuffer", mParams.frameDim.x * mParams.frameDim.y);
    }

    if (!mpBasePaths || mVarsChanged)
    {
        createBuffer(mpBasePaths, "basePaths", mParams.frameDim.x * mParams.frameDim.y * mStaticParams.numInitialSamples);
    }

    if (!mpBasePathsRad || mVarsChanged)
    {
        createBuffer(mpBasePathsRad, "basePathsRadiance", mParams.frameDim.x * mParams.frameDim.y * mStaticParams.numInitialSamples);
    }

    if (!mpDXShiftedPathsRad || mVarsChanged)
    {
        createBuffer(mpDXShiftedPathsRad, "basePathsRadiance",
                     mParams.frameDim.x * mParams.frameDim.y * mStaticParams.numInitialSamples * 2);
    }

    if (!mpDYShiftedPathsRad || mVarsChanged)
    {
        createBuffer(mpDYShiftedPathsRad, "basePathsRadiance",
                     mParams.frameDim.x * mParams.frameDim.y * mStaticParams.numInitialSamples * 2);
    }

    // Create reservoirs
    if (!mpPrimalPaths || mVarsChanged)
    {
        createBuffer(mpPrimalPaths, "basePaths", mParams.frameDim.x * mParams.frameDim.y * pathsPerPixel);
        printf("PathSampleRecord size: %d\n", mpPrimalPaths->getElementSize());
    }

    if (!mpPrimalReservoirs || mVarsChanged)
    {
        createBuffer(mpPrimalReservoirs, "primalReservoirs", mParams.frameDim.x * mParams.frameDim.y * pathsPerPixel);
        printf("PathReservoir size: %d\n", mpPrimalReservoirs->getElementSize());
    }

    if (!mpPrimalSampledTemporalReservoirs || mVarsChanged)
        createBuffer(mpPrimalSampledTemporalReservoirs, "temporalPrimalReservoirs", mParams.frameDim.x * mParams.frameDim.y);

    mpDXDiffIntegrator->prepareBuffers(mpReflectPass, mParams.frameDim, mVarsChanged);
    mpDYDiffIntegrator->prepareBuffers(mpReflectPass, mParams.frameDim, mVarsChanged);
}

void GPathTracer::prepareTracePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpGPathTracerBlock || mVarsChanged)
    {
        createParameterBlock(mpGPathTracerBlock, "gPathTracer");
    }

    if (!mpPrimalReSTIR || mVarsChanged)
    {
        createParameterBlock(mpPrimalReSTIR, "primalIntegrator");
    }

    if (!mpDXIntegrator || mVarsChanged)
    {
        createParameterBlock(mpDXIntegrator, "DXIntegrator");
    }

    if (!mpDYIntegrator || mVarsChanged)
    {
        createParameterBlock(mpDYIntegrator, "DYIntegrator");
    }

    // Primal Temporal Look For Neighbours pass
    auto primalTemporalLookForNeighboursRootVar = mpPrimalTemporalLookForNeighboursPass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(primalTemporalLookForNeighboursRootVar);
    }
    primalTemporalLookForNeighboursRootVar["gPathTracer"] = mpGPathTracerBlock;
    primalTemporalLookForNeighboursRootVar["params"].setBlob(mParams);
    primalTemporalLookForNeighboursRootVar["primalPaths"] = mpPrimalPaths;
    primalTemporalLookForNeighboursRootVar["primalReservoirs"] = mpPrimalReservoirs;
    primalTemporalLookForNeighboursRootVar["sampledTemporalReservoirs"] = mpPrimalSampledTemporalReservoirs;
    primalTemporalLookForNeighboursRootVar["pathsPerPixel"] = pathsPerPixel;
    primalTemporalLookForNeighboursRootVar["temporalFracPixelLocs"] = mpPrimalTemporalFracPixelLocs;
    primalTemporalLookForNeighboursRootVar["canFindTemporalNeighbours"] = mpPrimalCanFindTemporalNeighbors;
    mpScene->setRaytracingShaderData(pRenderContext, primalTemporalLookForNeighboursRootVar);
    mpPixelDebug->prepareProgram(mpPrimalTemporalLookForNeighboursPass->getProgram(), primalTemporalLookForNeighboursRootVar);

    // Primal Temporal Reuse pass
    auto primalTemporalReuseRootVar = mpPrimalTemporalReusePass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(primalTemporalReuseRootVar);
    }
    primalTemporalReuseRootVar["params"].setBlob(mParams);
    primalTemporalReuseRootVar["pathsPerPixel"] = pathsPerPixel;
    primalTemporalReuseRootVar["gPathTracer"] = mpGPathTracerBlock;
    primalTemporalReuseRootVar["temporalReservoirs"] = mpPrimalSampledTemporalReservoirs;
    primalTemporalReuseRootVar["primalPaths"] = mpPrimalPaths;
    primalTemporalReuseRootVar["curReservoirs"] = mpPrimalReservoirs;
    primalTemporalReuseRootVar["temporalFracPixelLocs"] = mpPrimalTemporalFracPixelLocs;
    primalTemporalReuseRootVar["basePaths"] = mpBasePaths;
    primalTemporalReuseRootVar["basePathsRadiance"] = mpBasePathsRad;
    primalTemporalReuseRootVar["canFindTemporalNeighbours"] = mpPrimalCanFindTemporalNeighbors;
    mpScene->setRaytracingShaderData(pRenderContext, primalTemporalReuseRootVar);
    mpPixelDebug->prepareProgram(mpPrimalTemporalReusePass->getProgram(), primalTemporalReuseRootVar);

    // Primal Spatial Reuse pass
    auto primalSpatialReuseRootVar = mpPrimalSpatialReusePass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(primalSpatialReuseRootVar);
    }
    primalSpatialReuseRootVar["primalReSTIRSampler"] = mpPrimalReSTIR;
    primalSpatialReuseRootVar["pathsPerPixel"] = pathsPerPixel;
    primalSpatialReuseRootVar["primalPaths"] = mpPrimalPaths;
    primalSpatialReuseRootVar["primalReservoirs"] = mpPrimalReservoirs;
    mpScene->setRaytracingShaderData(pRenderContext, primalSpatialReuseRootVar);
    mpPixelDebug->prepareProgram(mpPrimalSpatialReusePass->getProgram(), primalSpatialReuseRootVar);

    auto dxTemporalReuseRootVar = mpDXTemporalReusePass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(dxTemporalReuseRootVar);
    }
    dxTemporalReuseRootVar["gPathTracer"] = mpGPathTracerBlock;
    dxTemporalReuseRootVar["multiPointReSTIR"] = mpDXIntegrator;
    dxTemporalReuseRootVar["shiftedPathsRadiance"] = mpDXShiftedPathsRad;
    dxTemporalReuseRootVar["shiftDir"] = 0;
    dxTemporalReuseRootVar["pathsPerPixel"] = pathsPerPixel;
    dxTemporalReuseRootVar["neighbourPixels"] = mpDXSpatialNeighbours;
    mpScene->setRaytracingShaderData(pRenderContext, dxTemporalReuseRootVar);
    mpPixelDebug->prepareProgram(mpDXTemporalReusePass->getProgram(), dxTemporalReuseRootVar);

    auto dxTemporalNegReuseRootVar = mpDXTemporalNegReusePass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(dxTemporalNegReuseRootVar);
    }
    dxTemporalNegReuseRootVar["gPathTracer"] = mpGPathTracerBlock;
    dxTemporalNegReuseRootVar["multiPointReSTIR"] = mpDXIntegrator;
    dxTemporalNegReuseRootVar["shiftDir"] = 0;
    dxTemporalNegReuseRootVar["pathsPerPixel"] = pathsPerPixel;
    dxTemporalNegReuseRootVar["neighbourPixels"] = mpDXSpatialNeighbours;
    mpScene->setRaytracingShaderData(pRenderContext, dxTemporalNegReuseRootVar);
    mpPixelDebug->prepareProgram(mpDXTemporalNegReusePass->getProgram(), dxTemporalNegReuseRootVar);

    auto dxTemporalLookForNeighboursRootVar = mpDXTemporalLookForNeighboursPass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(dxTemporalLookForNeighboursRootVar);
    }
    dxTemporalLookForNeighboursRootVar["gPathTracer"] = mpGPathTracerBlock;
    dxTemporalLookForNeighboursRootVar["multiPointReSTIR"] = mpDXIntegrator;
    dxTemporalLookForNeighboursRootVar["shiftDir"] = 0;
    dxTemporalLookForNeighboursRootVar["pathsPerPixel"] = pathsPerPixel;
    mpPixelDebug->prepareProgram(mpDXTemporalLookForNeighboursPass->getProgram(), dxTemporalLookForNeighboursRootVar);

    auto dxSpatialReuseRootVar = mpDXSpatialReusePass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(dxSpatialReuseRootVar);
    }
    dxSpatialReuseRootVar["gPathTracer"] = mpGPathTracerBlock;
    dxSpatialReuseRootVar["multiPointReSTIR"] = mpDXIntegrator;
    dxSpatialReuseRootVar["shiftDir"] = 0;
    dxSpatialReuseRootVar["pathsPerPixel"] = pathsPerPixel;
    dxSpatialReuseRootVar["shouldSpatialReuse"] = renderData.getTexture(kDXSpatialReuse);
    dxSpatialReuseRootVar["neighbourPixels"] = mpDXSpatialNeighbours;
    mpScene->setRaytracingShaderData(pRenderContext, dxSpatialReuseRootVar);
    mpPixelDebug->prepareProgram(mpDXSpatialReusePass->getProgram(), dxSpatialReuseRootVar);

    auto dxSpatialNegReuseRootVar = mpDXSpatialNegReusePass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(dxSpatialNegReuseRootVar);
    }
    dxSpatialNegReuseRootVar["gPathTracer"] = mpGPathTracerBlock;
    dxSpatialNegReuseRootVar["multiPointReSTIR"] = mpDXIntegrator;
    dxSpatialNegReuseRootVar["shiftDir"] = 0;
    dxSpatialNegReuseRootVar["pathsPerPixel"] = pathsPerPixel;
    dxSpatialNegReuseRootVar["neighbourPixels"] = mpDXSpatialNeighbours;
    mpScene->setRaytracingShaderData(pRenderContext, dxSpatialNegReuseRootVar);
    mpPixelDebug->prepareProgram(mpDXSpatialNegReusePass->getProgram(), dxSpatialNegReuseRootVar);

    auto dxSpatialLookForNeighboursRootVar = mpDXSpatialLookForNeighboursPass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(dxSpatialLookForNeighboursRootVar);
    }
    dxSpatialLookForNeighboursRootVar["gPathTracer"] = mpGPathTracerBlock;
    dxSpatialLookForNeighboursRootVar["multiPointReSTIR"] = mpDXIntegrator;
    dxSpatialLookForNeighboursRootVar["shiftDir"] = 0;
    dxSpatialLookForNeighboursRootVar["pathsPerPixel"] = pathsPerPixel;
    dxSpatialLookForNeighboursRootVar["neighbourPixels"] = mpDXSpatialNeighbours;
    mpPixelDebug->prepareProgram(mpDXSpatialLookForNeighboursPass->getProgram(), dxSpatialLookForNeighboursRootVar);

    auto dyTemporalReuseRootVar = mpDYTemporalReusePass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(dyTemporalReuseRootVar);
    }
    dyTemporalReuseRootVar["gPathTracer"] = mpGPathTracerBlock;
    dyTemporalReuseRootVar["multiPointReSTIR"] = mpDYIntegrator;
    dyTemporalReuseRootVar["shiftedPathsRadiance"] = mpDYShiftedPathsRad;
    dyTemporalReuseRootVar["shiftDir"] = 1;
    dyTemporalReuseRootVar["pathsPerPixel"] = pathsPerPixel;
    dyTemporalReuseRootVar["neighbourPixels"] = mpDYSpatialNeighbours;
    mpScene->setRaytracingShaderData(pRenderContext, dyTemporalReuseRootVar);
    mpPixelDebug->prepareProgram(mpDYTemporalReusePass->getProgram(), dyTemporalReuseRootVar);

    auto dyTemporalNegReuseRootVar = mpDYTemporalNegReusePass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(dyTemporalNegReuseRootVar);
    }
    dyTemporalNegReuseRootVar["gPathTracer"] = mpGPathTracerBlock;
    dyTemporalNegReuseRootVar["multiPointReSTIR"] = mpDYIntegrator;
    dyTemporalNegReuseRootVar["shiftDir"] = 1;
    dyTemporalNegReuseRootVar["pathsPerPixel"] = pathsPerPixel;
    dyTemporalNegReuseRootVar["neighbourPixels"] = mpDYSpatialNeighbours;
    mpScene->setRaytracingShaderData(pRenderContext, dyTemporalNegReuseRootVar);
    mpPixelDebug->prepareProgram(mpDYTemporalNegReusePass->getProgram(), dyTemporalNegReuseRootVar);

    auto dyTemporalLookForNeighboursRootVar = mpDYTemporalLookForNeighboursPass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(dyTemporalLookForNeighboursRootVar);
    }
    dyTemporalLookForNeighboursRootVar["gPathTracer"] = mpGPathTracerBlock;
    dyTemporalLookForNeighboursRootVar["multiPointReSTIR"] = mpDYIntegrator;
    dyTemporalLookForNeighboursRootVar["shiftDir"] = 1;
    dyTemporalLookForNeighboursRootVar["pathsPerPixel"] = pathsPerPixel;
    mpPixelDebug->prepareProgram(mpDYTemporalLookForNeighboursPass->getProgram(), dyTemporalLookForNeighboursRootVar);

    auto dySpatialReuseRootVar = mpDYSpatialReusePass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(dySpatialReuseRootVar);
    }
    dySpatialReuseRootVar["gPathTracer"] = mpGPathTracerBlock;
    dySpatialReuseRootVar["multiPointReSTIR"] = mpDYIntegrator;
    dySpatialReuseRootVar["shiftDir"] = 1;
    dySpatialReuseRootVar["pathsPerPixel"] = pathsPerPixel;
    dySpatialReuseRootVar["shouldSpatialReuse"] = renderData.getTexture(kDYSpatialReuse);
    dySpatialReuseRootVar["neighbourPixels"] = mpDYSpatialNeighbours;
    mpScene->setRaytracingShaderData(pRenderContext, dySpatialReuseRootVar);
    mpPixelDebug->prepareProgram(mpDYSpatialReusePass->getProgram(), dySpatialReuseRootVar);

    auto dySpatialNegReuseRootVar = mpDYSpatialNegReusePass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(dySpatialNegReuseRootVar);
    }
    dySpatialNegReuseRootVar["gPathTracer"] = mpGPathTracerBlock;
    dySpatialNegReuseRootVar["multiPointReSTIR"] = mpDYIntegrator;
    dySpatialNegReuseRootVar["shiftDir"] = 1;
    dySpatialNegReuseRootVar["pathsPerPixel"] = pathsPerPixel;
    dySpatialNegReuseRootVar["neighbourPixels"] = mpDYSpatialNeighbours;
    mpScene->setRaytracingShaderData(pRenderContext, dySpatialNegReuseRootVar);
    mpPixelDebug->prepareProgram(mpDYSpatialNegReusePass->getProgram(), dySpatialNegReuseRootVar);

    auto dySpatialLookForNeighboursRootVar = mpDYSpatialLookForNeighboursPass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(dySpatialLookForNeighboursRootVar);
    }
    dySpatialLookForNeighboursRootVar["gPathTracer"] = mpGPathTracerBlock;
    dySpatialLookForNeighboursRootVar["multiPointReSTIR"] = mpDYIntegrator;
    dySpatialLookForNeighboursRootVar["shiftDir"] = 1;
    dySpatialLookForNeighboursRootVar["pathsPerPixel"] = pathsPerPixel;
    dySpatialLookForNeighboursRootVar["neighbourPixels"] = mpDYSpatialNeighbours;
    mpPixelDebug->prepareProgram(mpDYSpatialLookForNeighboursPass->getProgram(), dySpatialLookForNeighboursRootVar);

    auto evaluatePassVar = mpEvaluatePass->getRootVar();
    if (mVarsChanged)
    {
        mpSampleGenerator->bindShaderData(evaluatePassVar);
    }
    evaluatePassVar["gGPathTracer"] = mpGPathTracerBlock;
    evaluatePassVar["gPrimalIntegrator"] = mpPrimalReSTIR;
    evaluatePassVar["gDXIntegrator"] = mpDXIntegrator;
    evaluatePassVar["gDYIntegrator"] = mpDYIntegrator;
    mpScene->setRaytracingShaderData(pRenderContext, evaluatePassVar);
    evaluatePassVar["outputPrimal"] = renderData.getTexture(kOutputPrimal);
    evaluatePassVar["outputDX"] = renderData.getTexture(kOutputDX);
    evaluatePassVar["outputDY"] = renderData.getTexture(kOutputDY);

    auto pathTracerVar = mpGPathTracerBlock->getRootVar();
    pathTracerVar["params"].setBlob(mParams);
    if (mStaticParams.useNEE && mpEmissiveLigthSampler)
    {
        mpEmissiveLigthSampler->bindShaderData(pathTracerVar["emissiveLightSampler"]);
    }
    pathTracerVar["prevCameraData"].setBlob(mpPrevCameraData);
    pathTracerVar["vbuffer"] = mpVBuffer;
    pathTracerVar["prevVBuffer"] = mpPrevVBuffer;
    pathTracerVar["vbufferCenter"] = renderData.getTexture(kInputVBufferCenter);
    pathTracerVar["motionVectors"] = renderData.getTexture(kInputMotionVectors);
    pathTracerVar["subPixelUV"] = renderData.getTexture(kInputSubPixelUV);
    pathTracerVar["viewDir"] = renderData.getTexture(kInputViewDir);

    auto primalReSTIRVar = mpPrimalReSTIR->getRootVar();
    primalReSTIRVar["gPathTracer"] = mpGPathTracerBlock;
    primalReSTIRVar["neighborOffsets"] = mpNeighborOffsets;
    primalReSTIRVar["primaryEmission"] = mpPrimaryEmission;
    primalReSTIRVar["basePaths"] = mpBasePaths;
    primalReSTIRVar["primalReservoirs"] = mpPrimalReservoirs;

    auto dxIntegratorVar = mpDXIntegrator->getRootVar();
    dxIntegratorVar["gPathTracer"] = mpGPathTracerBlock;
    dxIntegratorVar["neighborOffsets"] = mpNeighborOffsets;
    dxIntegratorVar["primaryEmission"] = mpPrimaryEmission;
    dxIntegratorVar["basePaths"] = mpBasePaths;
    dxIntegratorVar["basePathsRadiance"] = mpBasePathsRad;
    dxIntegratorVar["primaryDepth"] = renderData.getTexture(kInputDepth);
    dxIntegratorVar["pathsPerPixel"] = pathsPerPixel;
    dxIntegratorVar["shiftDir"] = 0;
    dxIntegratorVar["primaryAlbedo"] = mpPrimaryAlbedo;
    dxIntegratorVar["primaryNormalAndDepth"] = mpPrimaryNormalAndDepth;

    auto dyIntegratorVar = mpDYIntegrator->getRootVar();
    dyIntegratorVar["gPathTracer"] = mpGPathTracerBlock;
    dyIntegratorVar["neighborOffsets"] = mpNeighborOffsets;
    dyIntegratorVar["primaryEmission"] = mpPrimaryEmission;
    dyIntegratorVar["basePaths"] = mpBasePaths;
    dyIntegratorVar["basePathsRadiance"] = mpBasePathsRad;
    dyIntegratorVar["primaryDepth"] = renderData.getTexture(kInputDepth);
    dyIntegratorVar["pathsPerPixel"] = pathsPerPixel;
    dyIntegratorVar["shiftDir"] = 1;
    dyIntegratorVar["primaryAlbedo"] = mpPrimaryAlbedo;
    dyIntegratorVar["primaryNormalAndDepth"] = mpPrimaryNormalAndDepth;
    mpDXDiffIntegrator->bindShaderData(dxIntegratorVar);
    mpDYDiffIntegrator->bindShaderData(dyIntegratorVar);
}

void GPathTracer::prepareReconstructionPass(RenderContext* pRenderContext,
                                            const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "Prepare poisson reconstruction pass.");

    auto rootVar = mpPoissonSolverPass->getRootVar();
    rootVar["primal"] = renderData.getTexture(kOutputPrimal);
    rootVar["DX"] = renderData.getTexture(kOutputDX);
    rootVar["DY"] = renderData.getTexture(kOutputDY);
    rootVar["reconstructedLast"] = mpReconstructedLast;
    rootVar["outputReconstructed"] = renderData.getTexture(kOutputColor);
    rootVar["CB"]["gFrameDim"] = mParams.frameDim;

    // In iteration 0, set the previous reconstruction to the input
    pRenderContext->copyResource(mpReconstructedLast.get(), renderData.getTexture(kOutputPrimal).get());
}

bool GPathTracer::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;

    lightingChanged |= prepareEmissiveLighting(pRenderContext);

    return lightingChanged;
}

bool GPathTracer::prepareEmissiveLighting(RenderContext* pRenderContext)
{
    bool emissiveLightingChanged = false;

    if (mpScene && mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
        // Check if the emissive light sampler should be updated
        if (!mpEmissiveLigthSampler)
        {
            mpScene->getLightCollection(pRenderContext);
            FALCOR_ASSERT(!mpEmissiveLigthSampler);
            switch(mStaticParams.emissiveLightSamplerType)
            {
            case EmissiveLightSamplerType::Uniform:
                mpEmissiveLigthSampler = std::make_unique<EmissiveUniformSampler>(pRenderContext, mpScene);
                break;
            case EmissiveLightSamplerType::Power:
                mpEmissiveLigthSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, mpScene);
                break;
            default:
                FALCOR_UNREACHABLE();
            }

            emissiveLightingChanged = true;
        }
    }
    else
    {
        if (mpEmissiveLigthSampler)
        {
            mpEmissiveLigthSampler = nullptr;
            emissiveLightingChanged = true;
        }
    }

    if (mpEmissiveLigthSampler)
    {
        emissiveLightingChanged |= mpEmissiveLigthSampler->update(pRenderContext);
        auto defines = mpEmissiveLigthSampler->getDefines();

        if (mpPrimalTemporalReusePass && mpPrimalTemporalReusePass->getProgram()->addDefines(defines))
        {
            emissiveLightingChanged = true;
        }

        if (mpPrimalSpatialReusePass && mpPrimalSpatialReusePass->getProgram()->addDefines(defines))
        {
            emissiveLightingChanged = true;
        }

        if (mpPrimaryEmissionPass && mpPrimaryEmissionPass->getProgram()->addDefines(defines))
        {
            emissiveLightingChanged = true;
        }

        if (mpPrepareBasePathsPass && mpPrepareBasePathsPass->getProgram()->addDefines(defines))
        {
            emissiveLightingChanged = true;
        }

        if (mpDXTemporalReusePass && mpDXTemporalReusePass->getProgram()->addDefines(defines))
        {
            emissiveLightingChanged = true;
        }

        if (mpDXTemporalNegReusePass && mpDXTemporalNegReusePass->getProgram()->addDefines(defines))
        {
            emissiveLightingChanged = true;
        }

        if (mpDXSpatialReusePass && mpDXSpatialReusePass->getProgram()->addDefines(defines))
        {
            emissiveLightingChanged = true;
        }

        if (mpDXSpatialNegReusePass && mpDXSpatialNegReusePass->getProgram()->addDefines(defines))
        {
            emissiveLightingChanged = true;
        }

        if (mpDYTemporalReusePass && mpDYTemporalReusePass->getProgram()->addDefines(defines))
        {
            emissiveLightingChanged = true;
        }

        if (mpDYTemporalNegReusePass && mpDYTemporalNegReusePass->getProgram()->addDefines(defines))
        {
            emissiveLightingChanged = true;
        }

        if (mpDYSpatialReusePass && mpDYSpatialReusePass->getProgram()->addDefines(defines))
        {
            emissiveLightingChanged = true;
        }

        if (mpDYSpatialNegReusePass && mpDYSpatialNegReusePass->getProgram()->addDefines(defines))
        {
            emissiveLightingChanged = true;
        }

        if (mpEvaluatePass && mpEvaluatePass->getProgram()->addDefines(defines))
        {
            emissiveLightingChanged = true;
        }
    }

    return emissiveLightingChanged;
}

void GPathTracer::updatePrograms()
{
    FALCOR_ASSERT(mpScene);

    if (mNeedsToRecompile == false)
        return;

    logDebug("GPathTracer::updatePrograms: Recompiling shaders.");
    auto defines = mStaticParams.getDefines(*this);
    auto globalTypeConformances = mpScene->getTypeConformances();

    // Create compute passes
    ProgramDesc baseDesc;
    baseDesc.addShaderModules(mpScene->getShaderModules());
    baseDesc.addTypeConformances(globalTypeConformances);

    if (!mpPoissonSolverPass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kReconstructPassShaderFile).csEntry("main");
        mpPoissonSolverPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    auto preparePass = [&](ref<ComputePass> pass)
    {
        // Note that we must use set instead of add defines to replace any stale state.
        pass->getProgram()->setDefines(defines);

        // Recreate program vars. This may trigger recompilation if needed.
        // Note that program versions are cached, so switching to a previously used specialization is faster.
        pass->setVars(nullptr);
    };

    if (!mpPrimaryEmissionPass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kPrimaryEmissionShaderFile).csEntry("main");
        mpPrimaryEmissionPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpPrepareBasePathsPass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kPrepareBasePathsShaderFile).csEntry("main");
        mpPrepareBasePathsPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpPrimalTemporalLookForNeighboursPass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kPrimalTemporalLookForNeighboursShaderFile).csEntry("main");
        mpPrimalTemporalLookForNeighboursPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpPrimalTemporalReusePass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kPrimalTemporalReusePassShaderFile).csEntry("main");
        mpPrimalTemporalReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpPrimalSpatialReusePass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kPrimalSpatialReusePassShaderFile).csEntry("main");
        mpPrimalSpatialReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpDXTemporalLookForNeighboursPass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiTemporalLookForNeighboursShaderFile).csEntry("main");
        mpDXTemporalLookForNeighboursPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpDXTemporalReusePass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiPointTemporalReuseShaderFile).csEntry("main");
        mpDXTemporalReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpDXTemporalNegReusePass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiPointTemporalNegReuseShaderFile).csEntry("main");
        mpDXTemporalNegReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpDXSpatialReusePass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiSpatialReuseShaderFile).csEntry("main");
        mpDXSpatialReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpDXSpatialNegReusePass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiSpatialNegReuseShaderFile).csEntry("main");
        mpDXSpatialNegReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpDXSpatialLookForNeighboursPass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiSpatialLookForNeighboursShaderFile).csEntry("main");
        mpDXSpatialLookForNeighboursPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpDYTemporalLookForNeighboursPass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiTemporalLookForNeighboursShaderFile).csEntry("main");
        mpDYTemporalLookForNeighboursPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpDYTemporalReusePass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiPointTemporalReuseShaderFile).csEntry("main");
        mpDYTemporalReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpDYTemporalNegReusePass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiPointTemporalNegReuseShaderFile).csEntry("main");
        mpDYTemporalNegReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpDYSpatialReusePass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiSpatialReuseShaderFile).csEntry("main");
        mpDYSpatialReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpDYSpatialNegReusePass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiSpatialNegReuseShaderFile).csEntry("main");
        mpDYSpatialNegReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpDYSpatialLookForNeighboursPass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiSpatialLookForNeighboursShaderFile).csEntry("main");
        mpDYSpatialLookForNeighboursPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpEvaluatePass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kEvaluatePassShaderFile).csEntry("main");
        mpEvaluatePass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpReflectPass || mNeedsToRecompile)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kReflectTypesShaderFile).csEntry("main");
        mpReflectPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    preparePass(mpPrimaryEmissionPass);
    preparePass(mpPrepareBasePathsPass);
    preparePass(mpPoissonSolverPass);
    preparePass(mpPrimalTemporalLookForNeighboursPass);
    preparePass(mpPrimalTemporalReusePass);
    preparePass(mpPrimalSpatialReusePass);
    preparePass(mpDXTemporalLookForNeighboursPass);
    preparePass(mpDXTemporalReusePass);
    preparePass(mpDXTemporalNegReusePass);
    preparePass(mpDXSpatialReusePass);
    preparePass(mpDXSpatialNegReusePass);
    preparePass(mpDXSpatialLookForNeighboursPass);
    preparePass(mpDYTemporalLookForNeighboursPass);
    preparePass(mpDYTemporalReusePass);
    preparePass(mpDYTemporalNegReusePass);
    preparePass(mpDYSpatialReusePass);
    preparePass(mpDYSpatialNegReusePass);
    preparePass(mpDYSpatialLookForNeighboursPass);
    preparePass(mpEvaluatePass);
    preparePass(mpReflectPass);

    // Need to re-bind parameter data.
    mpGPathTracerBlock = nullptr;

    mVarsChanged = true;
    mNeedsToRecompile = false;
}

void GPathTracer::prepareSampleGenerator(RenderContext* pRenderContext)
{
    if (!mpSampleGenerator)
    {
        mpSampleGenerator = SampleGenerator::create(mpDevice, mStaticParams.sampleGenerator);
        assert(mpSampleGenerator);
    }
}

void GPathTracer::prepareDiffIntegrator(RenderContext* pRenderContext)
{
    if (!mpDXDiffIntegrator)
    {
        mpDXDiffIntegrator = DiffIntegrator::create(mpDevice, mStaticParams.diffIntegrator);
        assert(mpDXDiffIntegrator);
    }

    if (!mpDYDiffIntegrator)
    {
        mpDYDiffIntegrator = DiffIntegrator::create(mpDevice, mStaticParams.diffIntegrator);
        assert(mpDYDiffIntegrator);
    }
}

void GPathTracer::setFrameDim(const uint2 frameDim)
{
    auto prevFrameDim = mParams.frameDim;
    mParams.frameDim = frameDim;

    if (any(mParams.frameDim != prevFrameDim))
    {
        mVarsChanged = true;
    }
}

bool GPathTracer::renderRenderingUI(Gui::Widgets& widget)
{
    bool valueChanged = false;

    // Have a static value for max samples for pixel.
    valueChanged |= widget.var("Samples/pixel", mStaticParams.samplesPerPixel,
                                1u, kMaxSamplesPerPixel);
    widget.tooltip("Number of samples per pixel.");

    valueChanged |= widget.var("Max bounces", mStaticParams.maxBounces, 0u,
                                kBounceLimit);
    widget.tooltip("Maximum number of surface bounces.");

    valueChanged |= widget.checkbox("Use alpha test", mStaticParams.useAlphaTest, true);
    widget.tooltip("Whether to use alpha test.");

    valueChanged |= widget.checkbox("Use NEE", mStaticParams.useNEE, false);
    widget.tooltip("Whether to use next event estimation.");

    valueChanged |= widget.var("Initial sample count", mStaticParams.numInitialSamples,
                               1u, kMaxNumInitialSamples);
    widget.tooltip("Number of initial candidate samples for RIS.");

    valueChanged |= widget.var("Shift mapping type", mStaticParams.shiftMappingType, 0u, 2u);
    widget.tooltip("Shift mapping type.");

    if (widget.dropdown("Sample generator", SampleGenerator::getGuiDropdownList(), mStaticParams.sampleGenerator))
    {
        mpSampleGenerator = SampleGenerator::create(mpDevice, mStaticParams.sampleGenerator);
        valueChanged = true;
    }

    if (mStaticParams.useNEE)
    {
        if (mpScene && mpScene->useEmissiveLights())
        {
            if (auto group = widget.group("Emissive Sampler"))
            {
                if (widget.dropdown("Emissive sampler", mStaticParams.emissiveLightSamplerType))
                {
                    valueChanged = true;
                }
                widget.tooltip("Select which emissive light sampler to use.");

                if (mpEmissiveLigthSampler)
                {
                    if (mpEmissiveLigthSampler->renderUI(group))
                    {
                        valueChanged = true;
                    }
                }
            }
        }
    }

    // Temporal Reuse
    if (auto group = widget.group("Temporal Reuse", true))
    {
        valueChanged |= group.checkbox("Enable", mStaticParams.useTemporalReuse, false);
        valueChanged |= group.var("History Length", mParams.historyLength, 4.0f, 1024.0f);
    }

    if (auto group = widget.group("Spatial Reuse", true))
    {
        valueChanged |= group.checkbox("Enable", mStaticParams.useSpatialReuse, false);
        valueChanged |= group.var("Spatial neighbours count", mStaticParams.numSpatialNeighbours, 1u, 16u);
    }

    widget.text("Poisson reconstruction", false);
    valueChanged |= widget.var("Alpha",
                                mStaticParams.poissonSolverAlpha,
                                0.0f, kMaxPoissonSolverAlpha);
    widget.tooltip("Alpha value for poisson reconstruction.");

    valueChanged |= widget.var("Max iterations",
                                mStaticParams.poissonSolverIterations,
                                4u, kMaxPoissonSolverIterations);
    widget.tooltip("Maximum number of iterations for poisson reconstruction.");

    return valueChanged;
}


void GPathTracer::createNeighborOffsetTexture(uint32_t sampleCount)
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

    mpNeighborOffsets = mpDevice->createTexture1D(sampleCount,
                                                  ResourceFormat::RG8Snorm,
                                                  1, 1, offsets.get());
}

void GPathTracer::create2DTexture(ref<Texture>& texture, uint2 dim, ResourceFormat format)
{
    texture = mpDevice->createTexture2D(dim.x,
                                        dim.y,
                                        format,
                                        1,
                                        1,
                                        nullptr,
                                        ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
}

void GPathTracer::createBuffer(ref<Buffer>& pBuffer, std::string reflectVarName, int requiredElementCount)
{
    pBuffer = mpDevice->createStructuredBuffer(mpReflectPass->getRootVar()[reflectVarName],
                                               requiredElementCount,
                                               ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                                               MemoryType::DeviceLocal,
                                               nullptr,
                                               false);
}

void GPathTracer::createParameterBlock(ref<ParameterBlock>& pParameterBlock,
                                       std::string reflectName)
{
    auto pReflection = mpReflectPass->getProgram()->getReflector();
    auto pBlockReflection = pReflection->getParameterBlock(reflectName);
    FALCOR_ASSERT(pBlockReflection);
    pParameterBlock = ParameterBlock::create(mpDevice, pBlockReflection);
    FALCOR_ASSERT(mpGPathTracerBlock);
    mVarsChanged = true;
}

DefineList GPathTracer::StaticParams::getDefines(const GPathTracer& owner) const
{
    DefineList defines;

    // RIS or PT settings
    defines.add("SAMPLES_PER_PIXEL", std::to_string(samplesPerPixel));
    defines.add("MAX_BOUNCES", std::to_string(maxBounces));
    defines.add("USE_ALPHA_TEST", useAlphaTest ? "1" : "0");
    defines.add("USE_NEE", std::to_string(useNEE));
    defines.add("NUM_INITIAL_SAMPLES", std::to_string(numInitialSamples));
    defines.add("USE_TEMPORAL_REUSE", std::to_string(useTemporalReuse));
    defines.add("USE_SPATIAL_REUSE", std::to_string(useSpatialReuse));
    defines.add("NUM_SPATIAL_NEIGHBOURS", std::to_string(numSpatialNeighbours));
    defines.add("NEIGHBOR_OFFSET_COUNT", std::to_string(kNeighborOffsetCount));
    defines.add("SHIFT_MAPPING_TYPE", std::to_string(shiftMappingType));

    // Poisson reconstruction
    defines.add("POISSON_SOLVER_ALPHA", std::to_string(poissonSolverAlpha));
    defines.add("POISSON_SOLVER_ITERATIONS", std::to_string(poissonSolverIterations));

    // Sample generator
    FALCOR_ASSERT(owner.mpSampleGenerator);
    defines.add(owner.mpSampleGenerator->getDefines());

    // Diff Integrator
    FALCOR_ASSERT(owner.mpDXDiffIntegrator);
    defines.add(owner.mpDXDiffIntegrator->getDefines());
    FALCOR_ASSERT(owner.mpDYDiffIntegrator);
    defines.add(owner.mpDYDiffIntegrator->getDefines());

    // Lighting
    if (owner.mpEmissiveLigthSampler)
    {
        defines.add(owner.mpEmissiveLigthSampler->getDefines());
    }

    // Scene-specific
    const auto& scene = owner.mpScene;
    if (scene)
        defines.add(scene->getSceneDefines());
    defines.add("USE_ENV_LIGHT", scene && scene->useEnvLight() ? "1" : "0");
    defines.add("USE_ANALYTIC_LIGHTS", scene && scene->useAnalyticLights() ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", scene && scene->useEmissiveLights() ? "1" : "0");

    return defines;
}
