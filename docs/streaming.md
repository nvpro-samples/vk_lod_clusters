# Streaming Operations

![image illustrating the streaming operations](lod_streaming.png)

The streaming system operates at the granularity of geometry groups. One group 
contains multiple clusters that were decimated together and are seamless among each other.

Each geometry has an array that stores the device address for a group,
`Geometry::streamingGroupAddresses`, this makes it easy to access the groups from
the LoD traversal nodes. The device address is legal only if the 64-bit value is
less than `STREAMING_INVALID_ADDRESS_BEGIN` (top most bit set). If it's invalid, 
than the lower 63-bits encode the frame index when it was last added to the request
load list, to prevent adding the same missing groups multiple times in a frame.

We differentiate between "active" groups, those that can be loaded and unloaded, and
"persistent" groups, that are always loaded.

Core files for the streaming system:
* [scene_streaming.hpp](../src/scene_streaming.hpp)
* [scene_streaming.cpp](../src/scene_streaming.cpp)
* [scene_streaming_utils.hpp](../src/scene_streaming_utils.hpp)
* [scene_streaming_utils.cpp](../src/scene_streaming_utils.cpp)
* [shaders/shaderio_streaming.h](../shaders/shaderio_streaming.h)

You will notice that the key components exist both on the C++ side as well as on
the device as `shaderio` structs. Each component can manage up to `STREAMING_MAX_ACTIVE_TASKS` tasks:
- `StreamingRequest`: Array of groups are missing and should be loaded or have not been accessed and can be unloaded. (purple in the diagram).
- `StreamingResident`: The table of resident geometry groups and clusters. The table might be filled sparsely; therefore, we keep a compact array of active group indices as well.
- `StreamingStorage`: Manages the storage and transfer for dynamically loaded geometry data, as well as freeing the memory for unloads. (dark red in diagram).
- `StreamingUpdate`: Defines the update on the device for the actual loads and unloads. We might request more than we can serve. It executes after the transfer of new geometry data is completed. (orange in the diagram)


In the UI under _"Streaming"_ one can change several behaviors and limitations.
These mostly drive how much streaming requests can be handled within a single frame,
and what the upper budgets for dynamic content are. These values do not represent 
recommendations and are just arbitrary defaults.


#### Initialization

For every geometry, the lowest level of detail group is uploaded and for ray tracing
the CLAS of all clusters within are generated once and also persistently stored.

The `Geometry::streamingGroupAddresses` are filled with appropriate addresses for
those persistently loaded groups, and the rest of the groups are set to be invalid.

The memory limits in the configuration do not cover this persistently loaded data,
which is always allocated.

However, when we register these persistent groups in the `StreamingResident` object table,
they do count against the limit of the table size. We automatically increase the
table size to at least have enough space to hold all low detail groups.

#### Runtime

The streaming system is frame-based. Each frame we trigger some tasks
and always initiate the streaming request task. There can be only one
task per kind applied on the device.

All these operations for a frame are configured within the `shaderio::SceneStreaming` struct that is filled
in `SceneStreaming::cmdBeginFrame` and accessible as both UBO and SSBO in the shaders.

We go through the core steps of streaming process in chronological order from the perspective of a request:

1. On the device we fill in the request task details.
   
   During traversal missing geometry groups are appended to the request
   load array.
   See `USE_STREAMING` within [shaders/traversal_run.comp.glsl](../shaders/traversal_run.comp.glsl),
   which is called by the renderer.
   After traversal any groups that have not been accessed in a while are 
   appended to the request unload array. 
   
   See [shaders/stream_agefilter_groups.comp.glsl](../shaders/stream_agefilter_groups.comp.glsl)
   which is called in `SceneStreaming::cmdPostTraversal`
   At the end of the frame we download the request to host in
   `SceneStreaming::cmdEndFrame`.

2. The request is handled on the host after checking its availability.
   
   The actual number of loads to perform is adjusted based on the available per-frame
   limits and if we can stay within the memory budget.
   The operation triggers the storage upload of newly loaded geometry groups via a `StreamingStorage` task.
   It also prepares a `StreamingUpdate` task, which encodes the patching of the scene
   and an update to the resident object table. Along with this is a `StreamingResident` task
   that provides the new state of active group indices.

   See `SceneStreaming::handleCompletedRequest`

3. Once the storage upload is completed, the appropriate update task is run.
   This update task actually patches the device side buffers so the loads
   and unloads become effective.
   When ray tracing is active, we will also build - on device - the CLAS of the newly loaded
   groups and handle their allocation management along with the patching.

   See [shaders/stream_update_scene.comp.glsl](../shaders/stream_update_scene.comp.glsl)
   run within `SceneStreaming::cmdPreTraversal`

   CLAS allocation management is done either through a persistent
   allocator system (`stream_allocator...` shader files) or through a simple
   compaction system (`stream_compaction...` shader files). More about that
   later.

4. After the update task is completed on the device the host can
   safely release the memory of unloaded groups. This memory is then
   recycled when we load new geometry groups at step (2).

   See the beginning of `SceneStreaming::cmdBeginFrame`.

This concludes the lifetime of a request from initial recording to all 
its dependent operations being completed.

Overall, both loading and unloading strategies are rather basic and there is room for improvement. 
Loading is purely based on the traversal, we expect that sorting the instances by camera distance 
and then seeding traversal nodes accordingly will help loading with priority around
the camera.

The streaming system has quite some configurable options, mostly balancing how 
many operations should be done within a single frame.
There is also the ability to use an asynchronous transfer queue for the data uploads,
otherwise we just upload on the main queue prior to the patch operations.

The provided defaults have not been tuned by any means and are not be seen as
recommendations.

Lastly, another major option is how the CLAS are allocated within the
fixed size CLAS buffer. Since the actual size of a CLAS is only
known on the device after it was built and the estimates from the host
can be a lot higher. We used solutions that can be implemented
on the device, not relying on further host readbacks but still
trying to make efficient use of the memory based on actual sizes.

Two options are provided, and they both first build new CLAS into
scratch space before moving them to their resident location.

- **Simple CLAS Compaction:**
  This simple scheme is based on a basic compaction algorithm that - on the device - 
  packs all resident cluster CLAS tightly before appending newly built ones.
  This can cause bursts of high amount of memory movement and a lot of bandwidth 
  and scratch space consumption. This is despite the fact that the new cluster
  API does provide functionality for moving objects to overlapping memory destinations.
  
  We do not recommend this, but it is the easiest way to get going.

  See `stream_compaction...` shader files

- **Persistent CLAS Allocator:**
  In this option we implement a persistent memory manager on the device
  so that CLAS-s are moved only once after initial building. See more
  [here](clas_allocation.md).