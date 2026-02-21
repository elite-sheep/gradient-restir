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
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

#ifndef HOST_CODE
__exported import Utils.Math.BitTricks;
#endif

#define SHIFT_MAPPING_RECONNECTION 0
#define SHIFT_MAPPING_RANDOM_REPLAY 1

/// Maximum supported sample count.
static const uint kMaxSamplesPerPixel = 16;
/// Maximum supported number of bounces per bounce category
static const uint kBounceLimit = 254;
/// Maximum iterations for poisson reconstruction
static const uint kMaxPoissonSolverIterations = 512;
/// Maximum alpha for poisson reconstruction
static const float kMaxPoissonSolverAlpha = 1.0;
/// Maximum supported initial sample count.
static const uint kMaxNumInitialSamples = 32;
/// Number of random numbers per hit
static const uint kNumOfSamplesPerHit = 16u;
/// Number of random numbers per NEE vertex
static const uint kNumOfSamplesPerNEEHit = 8u;

/*
G-PT parameters shared between host/device code.
*/
struct GPathTracerParams
{
    // Threshold for asserting on NaNs
    float assertThreshold = 1e9f;
    uint2 frameDim = { 0, 0 };
    uint _pad0;

    float historyLength = 32.0f;
    float spatialGatherRadius = 4.0f;
    float jacobianThreshold = 0.5f;
    uint  _pad5 = 0u;

    uint2 screenTiles = { 0, 0 };
    uint frameCount = 0;
    uint seed = 0;

    float sceneRadius;
    float distanceThreshold = 0.03f;
    float roughnessThreshold = 0.2f;
    uint _pad2;

    bool shouldRejectInnerShift = false;
    bool shouldRejectOuterShift = true;
    uint _pad3;
    uint _pad4;
};

END_NAMESPACE_FALCOR
