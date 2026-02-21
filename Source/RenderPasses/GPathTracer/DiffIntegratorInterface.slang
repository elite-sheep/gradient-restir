// Copyright @yucwang 2024

// The interface for diff integrator
interface IDiffIntegrator
{
    float4 evaluate(uint2 basePixel, uint2 shift);
}
