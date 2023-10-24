/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Utils.Attributes;

struct RetraceWorkloadQueue
{
    [root] RWByteAddressBuffer counter;
    RWByteAddressBuffer workload;
    RWByteAddressBuffer workloadExtra;

    uint addCounter(uint value)
    {
        uint sum = WaveActiveSum(value);
        uint offset = WavePrefixSum(value);
        uint originalValue;
        if (WaveIsFirstLane())
        {
            counter.InterlockedAdd(0, sum, originalValue);
        }
        originalValue = WaveReadLaneFirst(originalValue); 
        return originalValue + offset;
    }
}
