// Copyright @yucwang 2024

#include "DiffIntegratorType.slangh"

import DiffIntegrator;
import GPathTracer;
import PathShift;
import PathState;
import PrimalReSTIR;
import RISDataTypes;

RWStructuredBuffer<PathReservoir> primalReservoirs;
RWStructuredBuffer<PathReservoir> temporalPrimalReservoirs;
RWStructuredBuffer<PathReservoir> spatialPrimalReservoirs;
RWStructuredBuffer<float4> basePathsRadiance;

RWStructuredBuffer<VBuffer> vBuffer;

RWStructuredBuffer<PathSampleRecord> basePaths;
RWStructuredBuffer<PathReservoir> PosReservoirs;
RWStructuredBuffer<PathReservoir> TemporalPosReservoirs;
RWStructuredBuffer<PathReservoir> SpatialPosReservoirs;
RWStructuredBuffer<PathReservoir> NegReservoirs;
RWStructuredBuffer<PathReservoir> TemporalNegReservoirs;
RWStructuredBuffer<PathReservoir> SpatialNegReservoirs;

ParameterBlock<GPathTracer> gPathTracer;
ParameterBlock<PrimalReSTIR> primalIntegrator;
ParameterBlock<DiffIntegrator> DXIntegrator;
ParameterBlock<DiffIntegrator> DYIntegrator;

void main() {}
