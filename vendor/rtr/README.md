# vendor/rtr/

Thin wrappers that **link** the prebuilt [`real_time_registration`](https://github.com/xiandenis/real_time_registration) package.

## Consume (no sources in this workspace)

1. Outside this workspace, build/install RTR to an `install_party` prefix (example: `D:\work_space\real_time_registration\install_party`).
2. Sync into the LidarWorkFlow2 shared prefix:

```bat
REM from LidarWorkFlow2 root
sync-rtr-install.cmd
REM or: set RTR_INSTALL_PARTY=... && sync-rtr-install.cmd
```

3. Configure adapters / native ops with:

```bat
-DCMAKE_PREFIX_PATH=<LidarWorkFlow2>\install
```

Then `find_package(real_time_registration CONFIG REQUIRED)`.

## Operators (`CGRAPH_ADAPTERS_WITH_RTR`)

| op_id | RTR leaf | Notes |
|-------|----------|--------|
| `rtr.point_cloud.load` | `::point_cloud_io` | Path → `rtr.type.point_cloud` **File** realisation (Artifact DataRef) |
| `rtr.point_cloud.to_buffer` / `to_file` | `::point_cloud_io` + codec | Explicit File↔Buffer materialize (`rtr.codec.point_xyz_f32`) |
| `rtr.angle_downsample` | `AngleDownsample` (via initializer lib) | Angular depth pick; writes sidecar PCD |
| `rtr.voxel_medoid_downsample` | `::octree` `downsampleMedoid` | Hash-voxel true Medoid; writes sidecar PCD |
| `rtr.resolution_estimate` | `::resolution_estimate` | Emits `voxel_size` float (no cloud write) |
| `rtr.align_result.save` / `load` | `::registration_type` AlignResultIO | Value ↔ JSON file Artifact |
| `rtr.lidar_frame.save` / `load` | `::registration_type` LidarFrameIO | Isomorphic `rtr.type.lidar_frame` Value ↔ JSON (+ file Artifact) |
| `rtr.global_matrix.save` / `load` | `::global_matrix_file` | Station→global 4×4 table ↔ JSON |
| `rtr.registration_initializer` | `::registration_initializer` | Writes Root/subvoxel under `work_dir`; emits processed Artifact + voxel_size |
| `rtr.rough_global_reg` | `::rough_global_reg` | File Artifact in; typed `align_result` out |
| `rtr.fine_registration` | `::fine_registration` | Optional 4×4 `guess`, default I |
| `rtr.registration_report` | `::registration_report` | Two clouds + matrix → typed `registration_report` |
| `rtr.align_result.unpack` | (none) | Split AlignResult record fields |

**Payload model:** Point-cloud graph ports use **Value** Kind for both File (DataRef) and Buffer (Opaque) so `accepts=[file,buffer]` works. `cloud_port` accepts both; load/downsample **produce** File. Buffer uses `rtr.codec.point_xyz_f32` (LE `uint64` count + `count×3` float32 XYZ). Prefer `load_xyz_cloud_prefer_edge` when Frame+cloud may coexist (§9.3).

**Codec `rtr.codec.point_xyz_f32` layout:** little-endian; `uint64_t n`; then `n` triples of `float` (x,y,z). No pointers in Opaque.

**Frames:** `LidarFrame.global_matrix` ↔ `rtr.semantic.world_from_station` (`frame_from=station`, `frame_to=world`); helper `world_from_station_matrix_port`.

CMake: `-DCGRAPH_ADAPTERS_WITH_RTR=ON` (default). Plugin name: `cgraph_rtr.dll` / `cgraph_rtr.so`.
Do **not** wrap `RealTimeRegistrationSession` / `RealTimeManager` here.

`find_package(real_time_registration)` still requests Eigen 3.3; this tree may only have Eigen 5. `cmake/rtr-dep-shim/` plus a copy into CMake `pkgRedirects` satisfies that version check without linking Manager. Configure also needs PCL/OpenCV/PDAL from the prefix RTR was built against (example: `D:/install/vcpkg/installed/x64-windows` on `CMAKE_PREFIX_PATH`).

## Forbidden

- Do **not** copy the `real_time_registration` git tree into LidarWorkFlow2 / cgraph-adapters / cgraph-kernel / cgraph-studio.
- Do **not** vendor algorithm `.cpp` sources under `vendor/rtr/`.
- Do **not** treat in-workspace RTR source builds as the default dependency path.
- Do **not** link `real_time_manager` / umbrella `::real_time_registration`.
