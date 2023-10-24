/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"
__exported import StaticParams;
__exported import Rendering.ReSTIRPathTracing.PathReservoir;
__exported import Rendering.ReSTIRPathTracing.ReSTIRPT;

// maximum: 160 bytes
struct ReSTIRPTData
{
    PathRisState pathRis;
    PathRcInfo pathRcInfo;         
    float lastScatterPdf;
    float3 lastScatterWeight;
    float rrPdf;
    float pathTotalLength;

    __init()
    {
        pathRis = PathRisState();
        pathRcInfo = {};
        lastScatterPdf = 0.f;
        lastScatterWeight = 0.f;
        rrPdf = 1.f;
        pathTotalLength = 0.f;
    }
};
