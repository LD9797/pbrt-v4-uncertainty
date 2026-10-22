# NRC in pbrt-v4: the short version

This is the quick overview. For every detail (numbers, buffers, formulas, options) read [Detailed.md](Detailed.md).

## The idea

Following a light path for many bounces is slow and noisy.

So most paths are **stopped early** and we ask a small neural network: *"how much light leaves this point?"*
A few other paths are **traced all the way** for real, and their real answers **teach the network**.
The network learns while the image renders. This is the Neural Radiance Cache (NRC).

```
render path    (most pixels):  camera -> hit -> hit -> STOP -> ask the network -> add its answer to the pixel
training path  (few pixels):   camera -> hit -> hit -> keep going for real -> real answer -> teach the network
```

## The life of a ray, step by step

1. **Birth (`camera.cpp`).** The camera makes the ray. NRC decides: is this a *training path* or a *render path*? One pixel per small tile (starts 5 x 4) is a training path, and the chosen pixel moves every sample. After warmup, training paths only exist on every 4th sample. 1 in 16 training paths is an *RR-only* path (explained in step 6).
2. **Flight (`integrator.cpp`).** The ray hits a surface. If it is a light, its glow is added to the pixel like normal pbrt.
3. **Surface (`surfscatter.cpp`).** The material is evaluated and the next bounce direction is chosen like normal pbrt.
4. **Is this the query vertex? (`surfscatter.cpp`).** NRC measures how "blurry" the light has become (the *area spread*, from the paper by Mueller et al.). Once it is blurry enough, a network answer is good enough, so this hit is the **query vertex**. It is also the query vertex if the path reaches its maximum depth, the bounce fails, or Russian roulette kills it.
5. **Write down the point.** At the query vertex NRC stores 53 numbers describing it (position, direction, normal, roughness, reflectance, wavelengths), plus the reflectance and the light still left on the path (`beta`).
6. **Two different fates:**
   - **Render path, after warmup:** it **stops here**. No more rays. It waits for the network.
   - **Training path:** it **keeps going** for real. This extra tail is called the **suffix**. It is tracked from a fresh start (throughput 1). At each suffix vertex we store the vertex's numbers, the light picked up there (`local`) and how much of the next light survives the bounce (`step`). The tail stops when the path ends naturally, when it is blurry enough again, or after 4 vertices. If it was cut short, we ask the network for "the rest" (a *bootstrap*). An *RR-only* tail is never cut short: it runs until Russian roulette ends it, so its answer is fully honest and does not depend on the network.
7. **Network answers for render paths (`NRCInferenceForRenderPaths`).** All stopped paths are sent to the network at once. The pixel gets `beta * network_answer * reflectance`.
8. **Pixel is written (`UpdateFilm`).** Normal pbrt film work.
9. **Answers for training tails (`NRCTrainingSuffixFinish`).** We walk the tail **backward**: `light_here = local + step * light_of_next_vertex`. Each vertex of the tail becomes one teaching example (inputs + correct answer).
10. **Collect (`NRCAccumulateTrainingRecords`).** All examples are copied into one list on the GPU.
11. **Learn (`NRCTrainAccumulatedRecords`).** Once per sample, after all scanline bands: up to 65,536 examples are chosen (rare bright ones get a guaranteed share), split into 4 batches, and the network takes one training step per batch.
12. **Adjust.** The sample counter goes up. After 16 samples (`--nrc-warmup-samples`) the network is *warmed up*: from now on render paths use it. The tile size is changed a little so that each sample makes about 65,536 examples.

```
ONE SAMPLE

  for each band of scanlines:
      camera -> bounces (find query vertices) -> ask network (render paths)
             -> write pixels -> build tail answers -> collect examples

  then, once:  train the network   ->   warmup check   ->   adapt tile size
```

### Why a warmup?

An untrained network gives nonsense. For the first 16 samples we only **teach**; render paths are traced normally and tails are never cut short with a network guess. This prevents the network from learning from its own garbage.

### Why the reflectance trick?

The network learns "light arriving" (`q`). The surface colour (reflectance `R`) is multiplied back: `light = R * q`. This way the network does not have to relearn every material colour.

## What each file does

### camera.cpp

Creates the rays. Picks training paths (tile, every 4th sample, 1 in 16 RR-only). Saves the camera position for the area-spread maths.

```
camera.cpp
    |
    +-- create the normal pbrt camera ray
    +-- NRC: mark training path or render path
    +-- NRC: mark RR-only (1 in 16 training paths)
    +-- NRC: save the camera position (needed for area spread)
```

### film.cpp

Writes the pixel. It used to save the training target. It **no longer** does: targets now come from the tail's backward walk.

### integrator.cpp

The coordinator of the whole wavefront renderer. It:

- creates the network and all the `nrc*` buffers,
- runs the sample loop (see the diagram above),
- adds light (emission and light samples) to training tails,
- runs the NRC steps: `NRCInferenceForRenderPaths()`, `NRCTrainingSuffixFinish()`, `NRCAccumulateTrainingRecords()`, `NRCTrainAccumulatedRecords()` and `NRCUpdateTrainingTile()`,
- prints the debug output with `--nrc-debug`.

Before each band it calls `NRCResetSampleBuffers()` so the NRC state starts clean.

### surfscatter.cpp

Where a ray meets a surface. Normal pbrt job: evaluate the material, pick the next direction, shoot a light ray, create the next bounce ray.

NRC adds:

- the **area-spread** test (finds the query vertex),
- writing the **53 input numbers** of the vertex (`nrcInputs`, or `nrcSuffixInputs` for training tails),
- for **render paths after warmup**: stop the path (`nrcTerminateAndSubstitute`) and skip its own light sample and next ray,
- for **training paths**: keep going and fill the tail data (`local`, `step`, reflectance, luminance weights), and decide how the tail ends,
- remember the previous position and pdf for the next area-spread calculation.

## Summary

```
camera.cpp
-> make rays, choose training paths (tile, every 4th sample, 1 in 16 RR-only)

surfscatter.cpp
-> track area spread, find the query vertex, write the 53 inputs
-> stop render paths / continue training paths into a tail (suffix)

integrator.cpp
-> run the sample loop, ask the network for stopped paths,
   turn tails into examples (backward walk), train the network, adapt the tile

film.cpp
-> write the pixel (no longer builds training targets)
```

## What is done and what is left

Done:

- training tail (suffix) with its own area spread and 4-vertex cap,
- 1 in 16 RR-only tails,
- backward target walk (`Ls = local + step * Lnext`) and network bootstrap,
- reflectance split and `R * q` reconstruction,
- spectral (4 wavelength) input and output,
- tile size adaptation to about 65,536 examples per sample,
- GPU-side collecting and gathering of examples,
- bright-example guarantee in every training batch (`--nrc-brightness-oversampling=false` turns it off).

Left / to know:

- Training on every 4th sample after warmup is an experiment.
- `--nrc-train-steps` exists but is not used.
- Some materials (coated, hair, measured, ...) have no alpha/beta split yet.
- Small paper details (network size, weight averaging) may still differ.

See [Detailed.md](Detailed.md) for the full explanation, the option list and how to read `--nrc-debug` output.