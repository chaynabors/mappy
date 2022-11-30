# mappy
mappy is a suite of tools for parsing, compiling, and manipulating `.map` files

## parsing
Parsing logic for mappy is found within `mappy_parser` and can be used on it's own.
The parsing logic normalizes texture coordinate representation to the Valve 220 format for simplicity but leaves all other data untouched.

- [x] Standard map format
- [x] Valve 220 map format
- [ ] Quake 2 map format

## mappy
mappy transforms `.map` file entity definitions into a collection of entities consisting of the mesh representation of their halfspaces and their properties. This transformation is non-trivial and is done using a similar strategy to Quake. First a large cube is generated, with each halfspace clipping away at the cube until the final shape is left. Vertices, lines, and facets store an interned representation of their relationships as to generate vertex-index buffers for performant rendering. Facets are stored with their normals and their indices are ordered CCW by convention.

### roadmap
- Feature parity with old implementation
- Generate GI using photon mapping
- Parse properties given an `.fgd` file
- Content importer for bevy

## contributing
Please... help me
