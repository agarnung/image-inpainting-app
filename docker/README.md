# Docker instructions

The Qt 6 app ships as a Docker image. Build once, run whenever you want.

## Requirements

* Docker 20.10+ with Compose v2 (`docker compose`, not `docker-compose`) and **BuildKit enabled** (default on recent Docker).
* An X server reachable from the container:
  * **Linux** — your session already provides one.
  * **WSL2 (Windows)** — WSLg exposes one automatically. No extra setup.
  * Allow local Docker clients on first run: `xhost +local:docker` (the `start.sh` script does this for you).

## Quick start

From the project root:

```bash
cd docker
./start.sh
```

That script builds the image (if needed) and launches the app. The container is removed on exit.

Equivalent by hand:

```bash
cd docker
DOCKER_BUILDKIT=1 docker compose build
docker compose run --rm app
```

## Shared data folder

`docker/data/` on the host is bind-mounted to `/data` inside the container. It is also the default open/save directory in the app's file dialog (via the `APP_IMAGES_DIR` env variable), so:

* Drop the images you want to inpaint in `docker/data/` — they show up immediately when you click **Import image**.
* Exported images land in the same folder and are visible from the host.

## Rebuilding after code changes

Thanks to BuildKit cache mounts (apt + ccache) and the layered `COPY`, incremental rebuilds recompile only the `.cpp` files you touched — typically a few seconds.

```bash
docker compose build   # rebuilds just the changed layers
./start.sh             # or: docker compose run --rm app
```

For a fully clean build:

```bash
docker compose build --no-cache
```

## Display troubleshooting

The compose file sets `DISPLAY=${DISPLAY:-:0}`. If your X server listens on a different display, override it:

```bash
DISPLAY=:1 docker compose run --rm app
```

If the host's `DISPLAY` variable is set to something odd, unset it before running:

```bash
unset DISPLAY && ./start.sh
```

If Qt complains about the platform plugin, make sure the X socket is mounted (`/tmp/.X11-unix`, already in the compose file) and that `xhost +local:docker` has been run.

## Distributing the image

```bash
# Save to a .tar for offline transfer
docker save -o image-inpainting-app.tar image-inpainting-app:local

# On another machine
docker load -i image-inpainting-app.tar

# Or push to a registry
docker tag image-inpainting-app:local <user>/image-inpainting-app:<tag>
docker push <user>/image-inpainting-app:<tag>
```
