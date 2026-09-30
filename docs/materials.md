# Materials

The sample supports simple colored materials, alpha-masked materials, and (optionally) textured PBR shading.
However, the shading quality is kept rather basic given the focus was geometry in this sample. For higher quality
shading with a lot more features please refer to [vk_gltf_renderer](https://github.com/nvpro-samples/vk_gltf_renderer)
or [RTXMG SDK](https://github.com/NVIDIA-RTX/RTXMG).

## Textured PBR (metallic-roughness)

Textured PBR is **disabled by default**. When enabled, glTF **PBR metallic-roughness** materials can use base color, metallic-roughness, normal, occlusion, and emissive textures. **Specular-glossiness** materials are not supported for texturing (factor colors only).

**Command line:**

```text
--multimaterials 1 --attributes 7 --texturedmaterials 1
```

**UI:**

* _Scene Modifiers → Allow textured materials_
* _Scene Modifiers → Max texture MiB_ — VRAM budget for material textures (default 4096; 0 = no limit). Reloads textures when changed.
* _Cluster Settings → Other → Mesh Multi-Materials_
* _Cluster Settings → Other → Enabled Attributes_: enable **NRM**, **TAN**, and **TEX 0** (equivalent to `--attributes 7`)
* _Rendering Settings → Other → Facet shading_: disable it.

Changing _Allow textured materials_ reloads the scene, but does not require new processing.

**Restrictions:**

* **Workflow:** metallic-roughness only (no specular-glossiness texturing).
* **Vertex attributes:** meshes must provide `NORMAL`, `TANGENT`, and `TEXCOORD_0`. Tangents are required for normal mapping.
* **Texture coordinates:** only `TEXCOORD_0` is used; glTF per-texture texcoord indices are ignored.
* **File formats:** textures must be external `dds` or `ktx2` files (same as alpha-masked materials).
* **Loading:** material textures are loaded at scene init and are **not** streamed with geometry. By default a **4 GiB** VRAM budget applies (`--maxtexturemegabytes 4096`): textures start at full resolution, and if the total exceeds the budget, finer mips are dropped in round-robin order until the limit is met. Each texture always retains at least its coarsest mip. Set `--maxtexturemegabytes 0` to load all mips unconditionally.
* **Multi-material:** glTF meshes with multiple materials per mesh require `--multimaterials 1`.

Shaders are compiled with texture sampling only when the loaded scene actually contains textured materials (`HAS_TEXTURED_MATERIALS`).

## Alpha-masked materials

Alpha-masked materials work independently of textured PBR and remain enabled by default when the glTF uses `alphaMode = MASK`. Their textures are also loaded at scene init (not streamed) and count toward the texture VRAM budget.
