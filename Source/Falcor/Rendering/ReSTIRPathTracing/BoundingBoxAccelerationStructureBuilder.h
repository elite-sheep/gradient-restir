#pragma once

#include "Falcor.h"

namespace Falcor
{
    class BoundingBoxAccelerationStructureBuilder : public Object
    {
    public:
        static ref<BoundingBoxAccelerationStructureBuilder> Create(ref<Buffer> pBoundingBoxBuffer, ref<Device> pDevice);

        BoundingBoxAccelerationStructureBuilder(ref<Device> pDevice);

        void BuildAS(RenderContext* pContext, uint32_t boxCount, uint32_t rayTypeCount);

        void SetRaytracingShaderData(const ShaderVar& var, const std::string name, uint32_t rayTypeCount);

    private:

        void InitGeomDesc(uint32_t boxCount);

        void invalidateTlasCache();

        void BuildBlas(RenderContext* pContext);

        void FillInstanceDesc(std::vector<RtInstanceDesc>& instanceDescs, uint32_t rayCount, bool perMeshHitEntry);

        void BuildTlas(RenderContext* pContext, uint32_t rayCount, bool perMeshHitEntry);
        void BuildTlas2(RenderContext* pContext, uint32_t rayCount, bool perMeshHitEntry);

        ref<Buffer> m_BoundingBoxBuffer;

        struct BlasData
        {
            RtAccelerationStructurePrebuildInfo prebuildInfo = {};
            RtAccelerationStructureBuildInputs buildInputs = {};
            std::vector<RtGeometryDesc> geomDescs;

            uint64_t blasByteSize = 0;                      ///< Size of the final BLAS.
            uint64_t blasByteOffset = 0;                    ///< Offset into the BLAS buffer to where it is stored.
            uint64_t scratchByteOffset = 0;                 ///< Offset into the scratch buffer to use for rebuilds.
        };

        struct TlasData
        {
            ref<RtAccelerationStructure> pTlasObject;
            ref<Buffer> pTlasBuffer;
            ref<Buffer> pInstanceDescs;               ///< Buffer holding instance descs for the TLAS
        };

        std::vector<RtInstanceDesc> mInstanceDescs; ///< Shared between TLAS builds to avoid reallocating CPU memory
        std::unordered_map<uint32_t, TlasData> mTlasCache;  ///< Top Level Acceleration Structure for scene data cached per shader ray count
        ref<Buffer> mpTlasScratch;                    ///< Scratch buffer used for TLAS builds. Can be shared as long as instance desc count is the same, which for now it is.
        RtAccelerationStructurePrebuildInfo mTlasPrebuildInfo; ///< This can be reused as long as the number of instance descs doesn't change.

        std::vector<BlasData> mBlasData;    ///< All data related to the VPLs' BLASes.
        std::vector<ref<RtAccelerationStructure>> mBlasObjects; ///< BLAS API objects.

        bool mRebuildBlas = true;
        ref<Buffer> mpBlas;           ///< Buffer containing all BLASes.
        ref<Buffer> mpBlasScratch;    ///< Scratch buffer used for BLAS builds.

        ref<Device> mpDevice;
    };
}
