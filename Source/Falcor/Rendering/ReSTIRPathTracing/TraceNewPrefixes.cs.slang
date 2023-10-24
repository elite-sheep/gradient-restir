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

//import Scene.Raytracing;
//import Scene.RaytracingInline; // For visibility queries.
//import Scene.Intersection;
import Rendering.Utils.PixelStats;
// import RenderPasses.Shared.Denoising.NRDConstants;
// import Utils.Math.PackedFormats;
// import Utils.Math.FormatConversion;
// import PathTracer;
// import PathTracerNRD;
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
    bool traceRay(inout PathState path, const Ray ray, out HitInfo hit, out float hitT, bool isPrevFrame=false)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff, isPrevFrame);
    }
};

struct VisibilityQuery : PathTracer::IVisibilityQuery
{
    bool traceVisibilityRay(const Ray ray, bool isPrevFrame = false)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff, isPrevFrame);
    }
};

/** Simple scheduler (fullscreen compute pass) using TraceRayInline.
 */
struct Scheduler
{
    //StructuredBuffer<PrefixGBuffer> prefixGbuffer;
    //RWStructuredBuffer<PathReservoir> pathReservoirs;

    int integrationPrefixId;
    bool shouldGenerateSuffix;

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */

    // TODO: enable this only in an unbiased mode

    void tracePath(uint pathID)
    {
        PathState path;
        gPathTracer.generatePath(pathID, path, shouldGenerateSuffix);
        gPathTracer.setupPathLogging(path);

        // Note the primary miss has already been handled by the separate path generation pass
        // the path tracer runs first. Abort execution here to avoid double work.
        if (!path.isHit()) return;

        while (path.isActive())
        {
            if (path.isHit())
            {
                VisibilityQuery vq;
                gPathTracer.handleHit(path, vq);
                if (path.isTerminated()) break;

                ClosestHitQuery chq;
                gPathTracer.nextHit(path, chq, false);
            }
            else
            {
                gPathTracer.handleMiss(path);
            }
            // write to the prefix reservoir
        }

        // write to prefix reservoir
        if (gPathTracer.restirpt.subpathReuse && !path.restirPTData.pathRcInfo.pathFlags.isUserFlagSet()) // haven't been written
        {
            gPathTracer.writePrefixPathReservoir(path);
        }

        gPathTracer.writeSuffixReservoir(path);
    }

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        uint pathID = pixel.x | (pixel.y << 12) | (integrationPrefixId << 24);
        tracePath(pathID);
    }
}

[numthreads(8, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel >= gPathTracer.params.frameDim)) return;

    gScheduler.run(pixel);
}
