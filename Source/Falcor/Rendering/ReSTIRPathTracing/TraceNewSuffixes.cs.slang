/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "Scene/SceneDefines.slangh"

import Rendering.Utils.PixelStats;
import Utils.Debug.PixelDebug;

import Scene.RaytracingInline;
import PathTracer;
import LoadShadingData;
import Utils.Color.ColorHelpers;
import ReSTIRPT;


ParameterBlock<PathTracer> gPathTracer;
Scheduler gScheduler;

struct ClosestHitQuery : PathTracer::IClosestHitQuery
{
    bool traceRay(inout PathState path, const Ray ray, out HitInfo hit, out float hitT, bool isPrevFrame = false)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff, isPrevFrame);
    }
};

struct VisibilityQuery : PathTracer::IVisibilityQuery
{
    bool traceVisibilityRay(const Ray ray, bool isPrevFrame=false)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff, isPrevFrame);
    }
};

/** Simple scheduler (fullscreen compute pass) using TraceRayInline.
 */
struct Scheduler
{
    StructuredBuffer<PrefixGBuffer> prefixGbuffer;
    RWStructuredBuffer<PathReservoir> pathReservoirs;

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */

    // TODO: enable this only in an unbiased mode

    void tracePath(uint pathID, int2 pixel)
    {
        PathState path;

        // fetch from prefix G Buffer
        int offset = gPathTracer.params.getReservoirOffset(pixel);

        HitInfo prefixHit = prefixGbuffer[offset].hit;
        
        float3 wo = prefixGbuffer[offset].wo;

        if (!prefixHit.isValid()) return;

        int prefixLength = gPathTracer.restirpt.minimumPrefixLength;
        if (gPathTracer.restirpt.subpathSettings.adaptivePrefixLength)
            prefixLength = pathReservoirs[offset].pathFlags.prefixLength();


        ShadingData sd = loadShadingData(prefixHit, 0.f, -wo, false, gPathTracer.createTextureSampler(), false);
        gPathTracer.generatePathForSuffixGeneration(pathID, prefixHit, sd.computeRayOrigin(), path, prefixLength);


        if (prefixHit.isValid()) path.restirPTData.pathRcInfo.pathFlags.insertPathTreeLength(prefixLength);

        // store randomSeed for suffix
        gPathTracer.setupPathLogging(path);

        // add NEE contribution
        VisibilityQuery vq;

        gPathTracer.addNEEForPrefixLastVertex(path, sd, vq);
        

        uint suffixInitSeed = gPathTracer.getCurrentSeed(path.sg);
        // TODO: sample within specific lobe in the lobe-dependent prefix case

        const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, gPathTracer.createTextureSampler(), getMaterialInstanceHintsAssumeTriangle(false));
        bool valid = gPathTracer.generateScatterRayReSTIRPT(sd, mi, path, false);
        
        if (!valid || !path.isActive())
        {
            path.terminate();
        }
        else
        {
            while (path.isActive())
            {
                ClosestHitQuery chq;
                gPathTracer.nextHit(path, chq, false);

                if (path.isHit())
                {
                    VisibilityQuery vq;
                    gPathTracer.handleHit(path, vq);
                }
                else
                {
                    gPathTracer.handleMiss(path);
                }
            }
        }

        // initRandomSeed doesn't matter. won't be used
        float pHat = PathRisState::toScalar(path.restirPTData.pathRis.integrand);
        path.restirPTData.pathRis.weight = pHat == 0.f ? 0.f : path.restirPTData.pathRis.weight / pHat;

        path.restirPTData.pathRis.pathFlags.insertPathTreeLength(path.restirPTData.pathRcInfo.pathFlags.pathTreeLength());
        path.restirPTData.pathRis.pathFlags.insertPrefixLength(prefixLength);
        pathReservoirs[offset].transferRisStateNoInitRandomSeed(path.restirPTData.pathRis, suffixInitSeed);
    }

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        uint pathID = pixel.x | (pixel.y << 12);
        tracePath(pathID, pixel);
    }
}

[numthreads(8, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel >= gPathTracer.params.frameDim)) return;

    gScheduler.run(pixel);
}
