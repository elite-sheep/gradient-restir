// Copyright @yucwang 2024

#include "DiffIntegratorType.slangh"

__exported import DiffIntegratorInterface;
import Utils.Debug.PixelDebug;

#ifndef DIFF_INTEGRATOR_TYPE
import MultiPointReSTIR;
typedef MultiPointReSTIR DiffIntegrator;
#endif

#if defined(DIFF_INTEGRATOR_TYPE) && DIFF_INTEGRATOR_TYPE == DIFF_INTEGRATOR_MULTI_POINT_RESTIR
import MultiPointReSTIR;
typedef MultiPointReSTIR DiffIntegrator;
#endif

void DiffSampleInitialCandidates(DiffIntegrator diffIntegrator, uint2 basePixel, uint2 shift)
{
}

void DiffTemporalReuse(DiffIntegrator diffIntegrator, uint2 basePixel, uint2 shift)
{
#if DIFF_INTEGRATOR_TYPE == DIFF_INTEGRATOR_MULTI_POINT_RESTIR
    MultiPointTemporalReuse(diffIntegrator, basePixel, shift);
#endif
}

void DiffSpatialReuse(DiffIntegrator diffIntegrator, uint2 basePixel, uint2 shift)
{
#if DIFF_INTEGRATOR_TYPE == DIFF_INTEGRATOR_MULTI_POINT_RESTIR
    MultiPointSpatialReuse(diffIntegrator, basePixel, shift);
#endif
}
