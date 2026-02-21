// Copyright @yucwang 2024

#pragma once
#include "Falcor.h"
#include "DiffIntegratorType.slangh"
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Core/Program/DefineList.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/UI/Gui.h"
#include <functional>
#include <memory>

namespace Falcor
{
class RenderContext;

class DiffIntegrator : public Object
{
    FALCOR_OBJECT(DiffIntegrator)
public:
    virtual ~DiffIntegrator() = default;

    /**
     * Factory function for creating a pixel diff integrator of the specified type.
     * @param[in] pDevice GPU device.
     * @param[in] type The type of pixel diff integrator. See DiffIntegratorType.slangh.
     * @return New object, or throws an exception on error.
     */
    static ref<DiffIntegrator> create(ref<Device> pDevice, uint32_t type);

    /**
     * Get macro definitions for this diff integrator.
     * @return Macro definitions that must be set on the shader program that uses this diff integrator.
     */
    virtual DefineList getDefines() const;

    /**
     * Binds the data to a program vars object.
     * @param[in] pVars ProgramVars of the program to set data into.
     */
    virtual void bindShaderData(const ShaderVar& var) const;

    /**
     * Render the diff integrator's UI.
     */
    virtual void renderUI(Gui::Widgets& widget) {}

    /**
     * Begin a frame.
     * This should be called at the beginning of each frame for integrators that do extra setup for each frame.
     * @param[in] pRenderContext Render context.
     * @param[in] frameDim Current frame dimension.
     * @return Returns true if internal state has changed and bindShaderData() should be called before using the integrator.
     */
    virtual bool beginFrame(RenderContext* pRenderContext, const uint2& frameDim) { return false; }

    /**
     * End a frame.
     * This should be called at the end of each frame for integrators that do extra setup for each frame.
     * @param[in] pRenderContext Render context.
     * @param[in] pRenderOutput Rendered output.
     */
    virtual void endFrame(RenderContext* pRenderContext);

    /**
     * Returns a GUI dropdown list of all available pixel diff integrator types.
     */
    static const Gui::DropdownList& getGuiDropdownList();

    /**
     * Register a pixel diff integrator of the specified type.
     * @param[in] type The type of sample generator. See DiffIntegratorType.slangh.
     * @param[in] name Descriptive name used in the UI.
     * @param[in] createFunc Function to create an instance of the pixel diff integrator.
     */
    static void registerType(uint32_t type, const std::string& name, std::function<ref<DiffIntegrator>(ref<Device>)> createFunc);

    void prepareBuffers(ref<ComputePass>& reflectPass, uint2 count, bool force);

    void prepareSpatialReuse(RenderContext* pRenderContext);

    uint pathsPerPixel = 2; ///< Number of paths per pixel.

protected:
    DiffIntegrator(ref<Device> pDevice, uint32_t type) : mpDevice(pDevice), mType(type) {}

    ref<Device> mpDevice;
    const uint32_t mType; ///< Type of pixel diff integrator. See DiffIntegratorType.slangh.

private:
    // Multi point ReSTIR Data
    ref<Buffer> mpPosReservoirs;
    ref<Buffer> mpPosPaths;
    ref<Buffer> mpNegReservoirs;
    ref<Buffer> mpNegPaths;
    ref<Buffer> mpPosSampledTemporalReservoirs;
    ref<Buffer> mpNegSampledTemporalReservoirs;
    ref<Texture> mpPosTemporalFracLocs;
    ref<Texture> mpNegTemporalFracLocs;

private:
    void createMultiPointReSTIRReservoirs(ref<ComputePass>& reflectPass, uint2 count, bool force);
    void createBuffer(ref<Buffer>& pBuffer, ref<ComputePass>& reflectPass,
                      std::string reflectVarName, int requiredElementCount);
    void createTypedBuffer(ref<Buffer>& pBuffer, ResourceFormat format, int requiredElementCount);
    void create2DTexture(ref<Texture>& texture, uint2 dim, ResourceFormat format);
    /**
     * Register all basic pixel diff integrator types.
     */
    static void registerAll();

    friend struct RegisterDiffIntegrators;
};
} // namespace Falcor
