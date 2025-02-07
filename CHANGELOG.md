# Changelog for vk_lod_clusters

* 2025-2-7:
  * Added _"File > Save Cache"_ menu entry, as well as `-autosavecache 1` option. This allows to store the results of the lod cluster mesh processing into a file next to the original model.
    This allows speeding up future load times of the model a lot. See new notes in **Model processing** section of README
  * Improved warnings and some memory statistics.
  * Streaming geometry memory now guaranteed to stay within limit.
* 2025-1-30: Initial release