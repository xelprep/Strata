# Local Maxwell (sm_52) build — not upstream

This folder holds the changes that run Strata on an older NVIDIA Maxwell card
(compute capability 5.2). They are deliberately **not** upstreamed: they live
only on our fork (xelprep/Strata), tracked on `main`. Upstream never has a
`local/` path — checked 2026-10-31: no commit of Niko1221/Strata ever touched
one — so pulling upstream changes into the fork never conflicts with this
folder. One fork-only commit does end fast-forward syncs: from now on,
`git pull --ff-only upstream/main` stops working; take a real merge instead
(never conflicts, upstream has no `local/` paths), or rebase the fork commit
onto the new upstream:

    git fetch upstream
    git merge upstream/main            # merge commit, conflict-free
    git push origin main               # the fork carries upstream + local/

GitHub's own Sync-fork button no longer works (a fork with commits upstream
lacks cannot be synced from GitHub), so keep the fork synced this way. The
patch itself stays data: upstream still changing the patched files only
matters at image build — `patch -p1 --dry-run` is checked there, and the
regenerate recipe below is the way back if it ever fails.

## How it works

- `maxwell.patch` — the changes upstream still lacks (as of the pull on
  2026-10-31, v0.2.9): `src/core/device.cu` (the experimental build's runtime floor —
  upstream's #3218 put the sm_52 floor in CMake but left the runtime check at
  6.0, so a stock v0.2.9 engine built for 52 refuses itself at start) and
  `setup.py` (`sm60_card` must admit 52, or the Terminal's launch path calls a
  Maxwell card "not supported" — upstream's `maxwell_card`/`maxwell_note` only
  print a warning and stop). It carries a third file now, `CMakeLists.txt`:
  the 2026-11-01 sm_52 docker build died in `moe_ws_down.cu` with 400 ptxas
  errors, "Feature 'f16 arithemetic and compare instructions' requires .target
  sm_53 or higher" — the new 0.1.55 experimental build's weight-stationary
  experts emit packed half-precision PTX (`hfma2`/`hadd2`/`h2sum`), which the
  desktop Maxwell (sm_52, unlike sm_53) cannot assemble or execute. Upstream
  never compiled those kernels below sm_60, so this hit our build first. The
  patch's arch loop marks below-sm_60 CUDA builds (`set(_strata_ws_skip ON)`)
  and the three `moe_ws` blocks — kernels, `STRATA_PREFILL_WS`, tests — skip:
  `prefill.cpp` keeps its no-WS stubs, the experts go through MMQ / Q6K as on
  the reporter's 0.1.44.1 M6000 build, and the WS path asks for a cc 6.0 card
  at run time anyway. Pascal (60) and Volta (70) builds keep everything.
- `Dockerfile` — a port of upstream's **new** source-build Dockerfile (upstream
  rewrote it on 2026-10-26: three stages, the engine shipped as an engine
  component package in `/opt/strata/components/engine`, entrypoint moved to
  `docker/entrypoint.sh`, root `docker-entrypoint.sh` deleted). Maxwell tweaks
  kept: CUDA 12.9 base image, `CUDA_ARCHITECTURES=52` (upstream's own loop now
  turns `-DSTRATA_EXPERIMENTAL_SM60=ON` on by itself for archs below 75), the
  patch applied in stage 1 (engine sources) AND stage 2 (so the Terminal bundle
  carries the patched setup.py), `BUILD_VISION=0` as in upstream's CUDA 12 recipe.
  The old single-stage BUILD.json stamp flow is gone.

## Build (on the PC the card is in)

    git pull                              # origin = our fork: carries upstream + local/
    cp local/Dockerfile Dockerfile        # the local image layer
    # stage 2 of the port reads terminal-src/ (upstream's staged layout): the Terminal's
    # sources, the fork's committed local/ folder inside them - same file list as
    # upstream's docker/build.sh, no engine package needed (stage 1 builds it)
    mkdir -p terminal-src
    git ls-files -z -- packaging bin strata.py strata_control strata_cli serve requirements.txt requirements-monitor.txt docs/MANAGEMENT_PROTOCOL.md docs/TERMINAL_LAUNCH.md data bench/prompts/standard-v1 setup.py ref/load.py CMakeLists.txt ':(glob)tools/**/*.py' | tar --null -T - -cf - | tar -xf - -C terminal-src
    docker build -t strata-maxwell --build-arg CUDA_ARCHITECTURES=52 \
                 --build-arg STRATA_COMMIT=$(git rev-parse HEAD) .

Stage 1's COPY lists the engine's files (not `COPY . .`), so an edit to the Dockerfile, this
README, terminal-src staging or the Terminal stage keeps the compiled sm_52 engine in the
layer cache - only stage 2 runs again. A `git pull` that touches the engine sources
(naturally) recompiles it. If stage 2's apt stops on a network slip, the split install layers
let a rerun continue from the layer that already has python3/ca-certificates.

The Dockerfile applies `local/maxwell.patch` during the build, so the patch must
stay in sync with the checkout it applies to. If `patch -p1 < local/maxwell.patch`
fails during a build, upstream changed one of the patched files: regenerate the
patch against the new checkout:

    git checkout -- setup.py src/core/device.cu CMakeLists.txt   # if needed
    git --no-pager diff -- setup.py src/core/device.cu CMakeLists.txt > local/maxwell.patch

(Regenerate all three patched files — the Dockerfile changes live in
`local/Dockerfile` itself, not in the patch.)

## After an upstream pull

The engine comes from the docker build now (stage 1 packages it for the image),
not from a BUILD.json stamp: after a pull that touches the engine sources,
rebuild the image (`docker build` recompiles the sm_52 engine once); the model
files, the prepared pack and the install config on the `/data` volume are never
touched. A pull that only touches the Terminal (docs, packaging, serve/) keeps
stage 1 in the cache and reruns just the Terminal stage.

Checked against the 2026-10-31 pull (v0.2.9): the regenerated 81-line, three-file
patch applies with no rejected hunks (`patch -p1 --dry-run`). The card PC's build
of 2026-11-01 died in stage 1 with the `moe_ws_down.cu` ptxas errors above; the
CMake skip is that build's fix. Rerun the `docker build` on the card PC: it should
now reach stage 1's end (if more arch-52 compile errors show up, report them —
upstream's Maxwell note, docs/OLDER_GPUS.md #3218, invites running reports).
