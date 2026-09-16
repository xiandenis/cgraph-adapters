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

## Forbidden

- Do **not** copy the `real_time_registration` git tree into LidarWorkFlow2 / cgraph-adapters / cgraph-kernel / cgraph-studio.
- Do **not** vendor algorithm `.cpp` sources under `vendor/rtr/`.
- Do **not** treat in-workspace RTR source builds as the default dependency path.
