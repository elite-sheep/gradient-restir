// Copyright @yucwang 2024

#include "DiffIntegrator.h"

namespace Falcor
{
static std::map<uint32_t, std::function<ref<DiffIntegrator>(ref<Device>)>> sFactory;
static Gui::DropdownList sGuiDropdownList;

ref<DiffIntegrator> DiffIntegrator::create(ref<Device> pDevice, uint32_t type)
{
    if (auto it = sFactory.find(type); it != sFactory.end())
    {
        return it->second(pDevice);
    }
    else
    {
        FALCOR_THROW("Can't create diff integrator. Unknown type");
    }
}

DefineList DiffIntegrator::getDefines() const
{
    DefineList defines;
    defines.add("DIFF_INTEGRATOR_TYPE", std::to_string(mType));
    return defines;
}

const Gui::DropdownList& DiffIntegrator::getGuiDropdownList()
{
    return sGuiDropdownList;
}

void DiffIntegrator::registerType(uint32_t type, const std::string& name, std::function<ref<DiffIntegrator>(ref<Device>)> createFunc)
{
    sGuiDropdownList.push_back({type, name});
    sFactory[type] = createFunc;
}

void DiffIntegrator::registerAll()
{
    registerType(
        DIFF_INTEGRATOR_MULTI_POINT_RESTIR,
        "Multi point ReSTIR",
        [](ref<Device> pDevice) { return ref<DiffIntegrator>(new DiffIntegrator(pDevice, DIFF_INTEGRATOR_MULTI_POINT_RESTIR)); }
    );
}

void DiffIntegrator::endFrame(RenderContext* pRenderContext)
{
}

void DiffIntegrator::prepareSpatialReuse(RenderContext* pRenderContext)
{
}

void DiffIntegrator::bindShaderData(const ShaderVar& var) const
{
    var["pathsPerPixel"] = pathsPerPixel;
    var["posReservoirs"] = mpPosReservoirs;
    var["posPaths"] = mpPosPaths;
    var["posSampledTemporalReservoirs"] = mpPosSampledTemporalReservoirs;
    var["posTemporalFracPixelLocs"] = mpPosTemporalFracLocs;
    var["negReservoirs"] = mpNegReservoirs;
    var["negPaths"] = mpNegPaths;
    var["negSampledTemporalReservoirs"] = mpNegSampledTemporalReservoirs;
    var["negTemporalFracPixelLocs"] = mpNegTemporalFracLocs;
}

void DiffIntegrator::prepareBuffers(ref<ComputePass>& reflectPass, uint2 frameDim, bool force)
{
    createMultiPointReSTIRReservoirs(reflectPass, frameDim, force);
}

void DiffIntegrator::createMultiPointReSTIRReservoirs(ref<ComputePass>& reflectPass, uint2 frameDim, bool force)
{
    uint count = frameDim.x * frameDim.y;
    if (!mpPosReservoirs || force)
        createBuffer(mpPosReservoirs, reflectPass, "PosReservoirs", count * pathsPerPixel);
    if (!mpNegReservoirs || force)
        createBuffer(mpNegReservoirs, reflectPass, "NegReservoirs", count * pathsPerPixel);
    if (!mpPosPaths || force)
        createBuffer(mpPosPaths, reflectPass, "basePaths", count * pathsPerPixel);
    if (!mpNegPaths || force)
        createBuffer(mpNegPaths, reflectPass, "basePaths", count * pathsPerPixel);
    if (!mpPosSampledTemporalReservoirs || force)
        createBuffer(mpPosSampledTemporalReservoirs, reflectPass, "PosReservoirs", count);
    if (!mpNegSampledTemporalReservoirs || force)
        createBuffer(mpNegSampledTemporalReservoirs, reflectPass, "NegReservoirs", count);
    if (!mpPosTemporalFracLocs || force)
        create2DTexture(mpPosTemporalFracLocs, frameDim, ResourceFormat::RG32Float);
    if (!mpNegTemporalFracLocs || force)
        create2DTexture(mpNegTemporalFracLocs, frameDim, ResourceFormat::RG32Float);
}

void DiffIntegrator::create2DTexture(ref<Texture>& texture, uint2 dim, ResourceFormat format)
{
    texture = mpDevice->createTexture2D(dim.x,
                                        dim.y,
                                        format,
                                        1,
                                        1,
                                        nullptr,
                                        ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
}

void DiffIntegrator::createBuffer(ref<Buffer>& pBuffer, ref<ComputePass>& reflectPass, std::string reflectVarName, int requiredElementCount)
{
    pBuffer = mpDevice->createStructuredBuffer(reflectPass->getRootVar()[reflectVarName],
                                               requiredElementCount,
                                               ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                                               MemoryType::DeviceLocal,
                                               nullptr,
                                               false);
}

void DiffIntegrator::createTypedBuffer(ref<Buffer>& pBuffer, ResourceFormat format,
                                       int requiredElementCount)
{
    pBuffer = mpDevice->createTypedBuffer(format, requiredElementCount,
                                          ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                                          MemoryType::DeviceLocal,
                                          nullptr);
}

static struct RegisterDiffIntegrators
{
    RegisterDiffIntegrators() { DiffIntegrator::registerAll(); }
} sRegisterDiffIntegrators;

} // namespace Falcor
