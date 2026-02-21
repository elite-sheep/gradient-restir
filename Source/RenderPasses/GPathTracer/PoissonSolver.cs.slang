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

import StaticParams;
import Params;

cbuffer CB
{
    uint2 gFrameDim;
}
groupshared bool converged = false;

// Inputs
RWTexture2D<float4> primal;
RWTexture2D<float4> DX;
RWTexture2D<float4> DY;
RWTexture2D<float4> reconstructedLast;

// Outputs
RWTexture2D<float4> outputReconstructed;

float4 getDXPixel(uint2 pixel)
{
    if (all(pixel < gFrameDim) && all(pixel >= uint2(0, 0)))
    {
        return DX[pixel];
    }

    return float4(0.0f);
}

float4 getDYPixel(uint2 pixel)
{
    if (all(pixel < gFrameDim) && all(pixel >= uint2(0, 0)))
    {
        return DY[pixel];
    }

    return float4(0.0f);
}

float4 getLastIterPixel(uint2 pixel)
{
    if (all(pixel < gFrameDim) && all(pixel >= uint2(0, 0)))
    {
        return reconstructedLast[pixel];
    }

    return float4(0.0f);
}

float getPixelWeight(uint2 pixel)
{
    if (all(pixel < gFrameDim) && all(pixel >= uint2(0, 0)))
    {
        return 1.0;
    }
    return 0.0;
}

void flushBuffer(uint2 pixel)
{
    if (all(pixel < gFrameDim) && all(pixel >= uint2(0, 0)))
    {
        reconstructedLast[pixel] = outputReconstructed[pixel];
    }
}

void reconstructPoissonL2(uint2 pixel)
{
    if (all(pixel < gFrameDim))
    {
        float alpha = kPoissonSolverAlpha;
        float weight = 0.0f;
        float4 base = alpha * alpha * primal[pixel];
        weight += alpha * alpha;
        float4 gxx = getDXPixel(pixel + uint2(-1, 0)) - getDXPixel(pixel) +
                        getLastIterPixel(pixel + uint2(1, 0)) + getLastIterPixel(pixel + uint2(-1, 0));
        float4 gyy = getDYPixel(pixel + uint2(0, -1)) - getDYPixel(pixel) +
                        getLastIterPixel(pixel + uint2(0, 1)) + getLastIterPixel(pixel + uint2(0, -1));
        weight += getPixelWeight(pixel + uint2(1, 0)) + getPixelWeight(pixel + uint2(-1, 0));
        weight += getPixelWeight(pixel + uint2(0, -1)) + getPixelWeight(pixel + uint2(0, 1));

        outputReconstructed[pixel] = (base + gxx + gyy) / weight;
    }
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel < gFrameDim))
    {
        reconstructPoissonL2(dispatchThreadId.xy);
        flushBuffer(dispatchThreadId.xy);
    }
}
