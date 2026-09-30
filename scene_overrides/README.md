# scene_overrides

Per-scene setting overrides that ship with the sample rather than with the model.

When a scene is loaded, the app looks for `<stem of the loaded file>.cfg` in this
directory and, if found, runs it as a config file *after* the scene's own config
file (if any) and *before* the scene is actually loaded. Settings that influence
loading (cluster config, skip filters, budgets, ...) therefore take effect.

Examples:

| loaded file                    | override file used                 |
|--------------------------------|------------------------------------|
| `zorah_textured_public.v1.cfg` | `zorah_textured_public.v1.cfg`     |
| `bunny.gltf`                   | `bunny.cfg`                        |

An override must not change `--scene`; that setting is restored after parsing.

The directory is searched next to the executable in the same way as `shaders`:
the source tree for regular builds, `vk_lod_clusters_files/scene_overrides` and
the executable directory for install builds.
