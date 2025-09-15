# GPU-Driven CLAS Allocation

![image illustrating the streaming operations](lod_allocation.png)

The goal of the persistent CLAS allocator is to provide a persistent 
CLAS memory location with a fixed budget CLAS buffer. This means we need
to move the CLAS only once from its scratch space to a permanent location.
We later reclaim that memory when the group owning the CLAS is unloaded.

The implementation is completely on the device and does not require the host.
However, we need to read back the status of the free space to the host, so that
it can guarantee not to schedule newly loaded groups that are not guaranteed to fit.

Building the CLAS into scratch space first allows us to easily access the actual
size of the CLAS when making the allocation. While upper bounds can be
queried on the host, they are typically far from the real consumption,
and we want to benefit from tight packing.

The allocator represents the memory usage in a bit array based on the granularity of
CLAS sizes. As of writing the minimum granularity is 128 bytes and
can be increased further in the UI via _"Allocator granularity shift bits"_.
This granularity forms the basic "units" that the allocator operates in.
All sizes, offsets etc. are based on these units and they map to range of 
bits in the big array.

The bits are set during allocation, and cleared during deallocation. We allocate
on a per-group level and allocation sizes are at minimum 32 units.

We scan a sector of bits within a single subgroup to find free gaps.
The default number of sector bits is expressed in shifting the value of 32, i.e. `32 << 10`
 (_"Allocator sector shift bits"_ is set to 10 in UI). The free gaps are clamped in their 
size to the maximum group allocation size we can ever get. Which is computed
by `maximumClasSize * clustersPerGroup`. The former is queried from the driver
based on the maximum number of cluster triangles and vertices, the latter was a setting
of how we configured the cluster LoD builder.

Following operations are performed per frame:

1. If there is groups to be unloaded as part of the update task,
   then execute [shaders/stream_allocator_unload_groups.comp.glsl](../shaders/stream_allocator_unload_groups.comp.glsl)
   to clear the appropriate bits.
2. If there was unloading, or we do new loading of groups we need to build
   the list of free gaps that the allocator can use. This is done in a few steps. First, we run
   [shaders/stream_allocator_build_freegaps.comp.glsl](../shaders/stream_allocator_build_freegaps.comp.glsl)
   which finds the gaps in sector bits and writes them out in an unsorted fashion into
   `StreamingAllocator::freeGapsPos` and `StreamingAllocator::freeGapsSize`. We also 
   bump the histogram over the various gap sizes, `StreamingAllocator::freeSizeRanges[size].count`.
3. We reset a global gap count via using `STREAM_SETUP_ALLOCATOR_FREEINSERT` in [shaders/stream_setup.glsl](../shaders/stream_setup.comp.glsl)
4. Using the global gap counter and the `StreamingAllocator::freeSizeRanges[size].count` the offset
   `StreamingAllocator::freeSizeRanges[size].offset` is computed for each size within 
   [shaders/stream_allocator_setup_insertion.comp.glsl](../shaders/stream_allocator_setup_insertion.comp.glsl).
   The shader resets the per-size counts.
5. Now the free gaps are binned by their size into the per-size array ranges that were just computed.
   [shaders/stream_allocator_freegaps_insert.comp.glsl](../shaders/stream_allocator_freegaps_insert.comp.glsl) is 
   responsible for this operation.
6. Finally, we have all the data to do the allocation of newly loaded groups.
   Details can be found in [shaders/stream_allocator_load_groups.comp.glsl](../shaders/stream_allocator_load_groups.comp.glsl).
   We compute the group's required allocation size from its CLAS sizes and then look for 
   free gaps of the same size or slightly bigger.
   When nothing is found, we will attempt to make bigger allocations combining multiple groups that didn't find a gap.
   Last but not least we will sub-allocate from the worst-case sized allocation gaps. We have guaranteed on the host
   that we would never trigger more loads than we have worst-case free space for.
7. To ensure this guarantee, after the allocation is completed, we store the state of the worst-case gap sizes that are left into the 
   currently recorded request task information.
   This is done by running `STREAM_SETUP_ALLOCATOR_STATUS` in [shaders/stream_setup.glsl](../shaders/stream_setup.comp.glsl)

In future versions we will try to optimize this scheme a bit further.