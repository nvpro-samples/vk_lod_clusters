#!/usr/bin/env python3
"""Generates the materialtest fixture: materialtest.gltf, materialtest.bin and
materialtest_alpha.ktx.

The fixture is a row of subdivided boxes that between them cover the mesh and material
states the sample branches on - vertex attribute sets, alpha masking, alpha blending,
double sided, several materials on one mesh, and a primitive with no material at all.
See tests/README.md.

Only run this when the fixture needs to change; the generated files are committed.

    python tests/make_materialtest.py
"""

import base64
import json
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))

# subdivisions per box face. Enough triangles that the lod build has something to do
# (6 * SUBDIV^2 * 2 triangles per box) without making the fixture large.
SUBDIV = 8

# face basis: origin corner, edge vectors, normal
FACES = [
    ((-1, -1, +1), (2, 0, 0), (0, 2, 0), (0, 0, +1)),  # +Z
    ((+1, -1, -1), (-2, 0, 0), (0, 2, 0), (0, 0, -1)),  # -Z
    ((+1, -1, +1), (0, 0, -2), (0, 2, 0), (+1, 0, 0)),  # +X
    ((-1, -1, -1), (0, 0, 2), (0, 2, 0), (-1, 0, 0)),  # -X
    ((-1, +1, +1), (2, 0, 0), (0, 0, -2), (0, +1, 0)),  # +Y
    ((-1, -1, -1), (2, 0, 0), (0, 0, 2), (0, -1, 0)),  # -Y
]


def normalize(v):
    length = sum(c * c for c in v) ** 0.5
    return tuple(c / length for c in v)


def build_box():
    """Returns (positions, normals, uv0, uv1, tangents, indices) for a subdivided cube."""
    positions, normals, uv0, uv1, tangents, indices = [], [], [], [], [], []

    for origin, edge_u, edge_v, normal in FACES:
        base = len(positions)
        tangent = normalize(edge_u)

        for iv in range(SUBDIV + 1):
            for iu in range(SUBDIV + 1):
                u = iu / SUBDIV
                v = iv / SUBDIV
                positions.append(tuple(origin[c] + edge_u[c] * u + edge_v[c] * v for c in range(3)))
                normals.append(normal)
                uv0.append((u, v))
                # deliberately different from uv0, so a second texcoord set is not a
                # duplicate that could be optimized away without anyone noticing
                uv1.append((v, 1.0 - u))
                tangents.append((tangent[0], tangent[1], tangent[2], 1.0))

        for iv in range(SUBDIV):
            for iu in range(SUBDIV):
                i0 = base + iv * (SUBDIV + 1) + iu
                i1 = i0 + 1
                i2 = i0 + (SUBDIV + 1)
                i3 = i2 + 1
                # counter clockwise seen from along the face normal, the basis vectors
                # above are picked so that edge_u x edge_v == normal for every face
                indices += [i0, i1, i2, i1, i3, i2]

    return positions, normals, uv0, uv1, tangents, indices


class BufferBuilder:
    """Accumulates accessor data into one buffer, 4 byte aligned."""

    def __init__(self):
        self.data = bytearray()
        self.views = []
        self.accessors = []

    def _view(self, payload, target):
        while len(self.data) % 4:
            self.data.append(0)
        offset = len(self.data)
        self.data += payload
        self.views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload), "target": target})
        return len(self.views) - 1

    def add_vec(self, values, components, name):
        payload = bytearray()
        for value in values:
            payload += struct.pack("<%df" % components, *value)
        view = self._view(payload, 34962)  # ARRAY_BUFFER

        flat = [value[c] for value in values for c in range(components)]
        mins = [min(flat[c::components]) for c in range(components)]
        maxs = [max(flat[c::components]) for c in range(components)]

        self.accessors.append({
            "bufferView": view,
            "componentType": 5126,  # FLOAT
            "count": len(values),
            "type": {2: "VEC2", 3: "VEC3", 4: "VEC4"}[components],
            "min": mins,
            "max": maxs,
            "name": name,
        })
        return len(self.accessors) - 1

    def add_indices(self, indices, name, first=0, count=None):
        count = len(indices) if count is None else count
        payload = struct.pack("<%dI" % len(indices), *indices)
        view = self._view(payload, 34963)  # ELEMENT_ARRAY_BUFFER
        self.accessors.append({
            "bufferView": view,
            "componentType": 5125,  # UNSIGNED_INT
            "count": count,
            "byteOffset": first * 4,
            "type": "SCALAR",
            "name": name,
        })
        return len(self.accessors) - 1

    def add_index_range(self, view_accessor, first, count, name):
        """A second accessor over the index buffer view of `view_accessor`."""
        source = self.accessors[view_accessor]
        self.accessors.append({
            "bufferView": source["bufferView"],
            "componentType": 5125,
            "count": count,
            "byteOffset": first * 4,
            "type": "SCALAR",
            "name": name,
        })
        return len(self.accessors) - 1


def write_ktx1_rgba(path, size, mip_levels, pixel_fn):
    """Minimal KTX1 writer, GL_SRGB8_ALPHA8, with a full mip chain.

    The sample only reads DDS and KTX, so the alpha mask material needs one of those -
    and it needs real mips, since those are uploaded per level.
    """
    levels = []
    level_size = size
    pixels = [[pixel_fn(x, y, size) for x in range(size)] for y in range(size)]

    for level in range(mip_levels):
        payload = bytearray()
        for row in pixels:
            for pixel in row:
                payload += bytes(pixel)
        levels.append((level_size, bytes(payload)))

        if level_size == 1:
            break
        # box filter down for the next level
        half = level_size // 2
        reduced = []
        for y in range(half):
            row = []
            for x in range(half):
                samples = [pixels[2 * y][2 * x], pixels[2 * y][2 * x + 1],
                           pixels[2 * y + 1][2 * x], pixels[2 * y + 1][2 * x + 1]]
                row.append(tuple(sum(s[c] for s in samples) // 4 for c in range(4)))
            reduced.append(row)
        pixels = reduced
        level_size = half

    header = bytes([0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A])
    header += struct.pack(
        "<13I",
        0x04030201,  # endianness
        0x1401,      # glType          GL_UNSIGNED_BYTE
        1,           # glTypeSize
        0x1908,      # glFormat        GL_RGBA
        0x8C43,      # glInternalFormat GL_SRGB8_ALPHA8
        0x1908,      # glBaseInternalFormat GL_RGBA
        size,        # pixelWidth
        size,        # pixelHeight
        0,           # pixelDepth
        0,           # numberOfArrayElements
        1,           # numberOfFaces
        len(levels), # numberOfMipmapLevels
        0,           # bytesOfKeyValueData
    )

    body = bytearray()
    for _, payload in levels:
        body += struct.pack("<I", len(payload))
        body += payload
        while len(body) % 4:
            body.append(0)

    with open(path, "wb") as f:
        f.write(header)
        f.write(bytes(body))
    return len(levels)


def main():
    positions, normals, uv0, uv1, tangents, indices = build_box()
    triangle_count = len(indices) // 3

    buf = BufferBuilder()
    a_pos = buf.add_vec(positions, 3, "POSITION")
    a_nrm = buf.add_vec(normals, 3, "NORMAL")
    a_uv0 = buf.add_vec(uv0, 2, "TEXCOORD_0")
    a_uv1 = buf.add_vec(uv1, 2, "TEXCOORD_1")
    a_tan = buf.add_vec(tangents, 4, "TANGENT")
    a_idx = buf.add_indices(indices, "indices")

    # the multi material mesh splits the same index buffer into two primitives
    half = (triangle_count // 2) * 3
    a_idx_lo = buf.add_index_range(a_idx, 0, half, "indices_lo")
    a_idx_hi = buf.add_index_range(a_idx, half, len(indices) - half, "indices_hi")

    full = {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv0, "TEXCOORD_1": a_uv1, "TANGENT": a_tan}
    uv0_only = {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv0}
    pos_nrm = {"POSITION": a_pos, "NORMAL": a_nrm}

    materials = [
        {"name": "Opaque", "pbrMetallicRoughness": {
            "baseColorFactor": [0.8, 0.2, 0.2, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.6},
         "alphaMode": "OPAQUE"},
        {"name": "TwoSided", "pbrMetallicRoughness": {
            "baseColorFactor": [0.2, 0.8, 0.3, 1.0], "metallicFactor": 0.1, "roughnessFactor": 0.4},
         "alphaMode": "OPAQUE", "doubleSided": True},
        # alpha masking needs a base color texture: the sample takes the mask from it and
        # falls back to opaque when the material has none
        {"name": "AlphaMask", "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0], "baseColorTexture": {"index": 0},
            "metallicFactor": 0.0, "roughnessFactor": 0.8},
         "alphaMode": "MASK", "alphaCutoff": 0.5, "doubleSided": True},
        {"name": "AlphaBlend", "pbrMetallicRoughness": {
            "baseColorFactor": [0.2, 0.4, 0.9, 0.45], "metallicFactor": 0.0, "roughnessFactor": 0.3},
         "alphaMode": "BLEND"},
        {"name": "Metal", "pbrMetallicRoughness": {
            "baseColorFactor": [0.9, 0.8, 0.4, 1.0], "metallicFactor": 1.0, "roughnessFactor": 0.15},
         "alphaMode": "OPAQUE"},
        {"name": "Emissive", "pbrMetallicRoughness": {
            "baseColorFactor": [0.1, 0.1, 0.1, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0},
         "emissiveFactor": [0.9, 0.5, 0.1], "alphaMode": "OPAQUE"},
    ]
    MAT_OPAQUE, MAT_TWOSIDED, MAT_MASK, MAT_BLEND, MAT_METAL, MAT_EMISSIVE = range(6)

    meshes = [
        # full attribute set, one material
        {"name": "BoxOpaqueFull", "primitives": [
            {"attributes": full, "indices": a_idx, "material": MAT_OPAQUE}]},
        # double sided, which flips back face culling and the ray flags
        {"name": "BoxTwoSided", "primitives": [
            {"attributes": full, "indices": a_idx, "material": MAT_TWOSIDED}]},
        # alpha masked, which pulls in the any hit / alpha test shader permutation
        {"name": "BoxAlphaMask", "primitives": [
            {"attributes": full, "indices": a_idx, "material": MAT_MASK}]},
        # alpha blended, skipped by default (--skipalphablended)
        {"name": "BoxAlphaBlend", "primitives": [
            {"attributes": full, "indices": a_idx, "material": MAT_BLEND}]},
        # two materials on one mesh, for --multimaterials and the per triangle material
        # weight in the lod build
        {"name": "BoxMultiMaterial", "primitives": [
            {"attributes": full, "indices": a_idx_lo, "material": MAT_METAL},
            {"attributes": full, "indices": a_idx_hi, "material": MAT_EMISSIVE}]},
        # three materials, one of them alpha masked, so a single mesh mixes geometry flags
        {"name": "BoxMixedAlpha", "primitives": [
            {"attributes": full, "indices": a_idx_lo, "material": MAT_MASK},
            {"attributes": full, "indices": a_idx_hi, "material": MAT_TWOSIDED}]},
        # reduced attribute sets: the lod build lays the attributes out from what is present
        {"name": "BoxTexcoord0Only", "primitives": [
            {"attributes": uv0_only, "indices": a_idx, "material": MAT_OPAQUE}]},
        {"name": "BoxPositionNormal", "primitives": [
            {"attributes": pos_nrm, "indices": a_idx, "material": MAT_METAL}]},
        # no material at all, which falls back to the default slot
        {"name": "BoxNoMaterial", "primitives": [
            {"attributes": pos_nrm, "indices": a_idx}]},
    ]

    spacing = 3.0
    nodes = []
    for i, mesh in enumerate(meshes):
        x = (i - (len(meshes) - 1) / 2.0) * spacing
        nodes.append({"name": mesh["name"], "mesh": i, "translation": [x, 0.0, 0.0]})
    # one mesh instanced a second time, so the scene is not purely one instance per geometry
    nodes.append({"name": "BoxOpaqueFull.001", "mesh": 0, "translation": [0.0, spacing, 0.0]})
    nodes.append({"name": "BoxAlphaMask.001", "mesh": 2, "translation": [spacing, spacing, 0.0]})

    # 16x16 keeps the committed binary at about 1.5 KB and still gives a 5 level mip chain,
    # which is what makes a mip upload that stops after level 0 visible
    mip_levels = write_ktx1_rgba(
        os.path.join(HERE, "materialtest_alpha.ktx"), 16, 16,
        # a 4x4 block checkerboard in alpha, so the mask punches holes that are obvious at a
        # glance, over an rg gradient so a wrong swizzle or a wrong mip is too
        lambda x, y, size: (
            int(255 * x / (size - 1)),
            int(255 * y / (size - 1)),
            120,
            255 if ((x // 4) + (y // 4)) % 2 == 0 else 0,
        ))

    gltf = {
        "asset": {"version": "2.0", "generator": "tests/make_materialtest.py"},
        "scene": 0,
        "scenes": [{"name": "MaterialTest", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "textures": [{"source": 0}],
        "images": [{"uri": "materialtest_alpha.ktx", "mimeType": "image/ktx"}],
        "accessors": buf.accessors,
        "bufferViews": buf.views,
        "buffers": [{"uri": "materialtest.bin", "byteLength": len(buf.data)}],
    }

    with open(os.path.join(HERE, "materialtest.bin"), "wb") as f:
        f.write(bytes(buf.data))
    with open(os.path.join(HERE, "materialtest.gltf"), "w", newline="\r\n") as f:
        json.dump(gltf, f, indent=1)
        f.write("\n")

    print("materialtest.gltf  %d meshes, %d nodes, %d materials" % (len(meshes), len(nodes), len(materials)))
    print("materialtest.bin   %d bytes, %d triangles per box" % (len(buf.data), triangle_count))
    print("materialtest_alpha.ktx  16x16 SRGB8_ALPHA8, %d mips" % mip_levels)


if __name__ == "__main__":
    main()
