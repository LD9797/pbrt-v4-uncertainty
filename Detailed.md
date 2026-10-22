# Neural Radiance Cache (NRC) in pbrt-v4: the detailed guide

This file explains, in plain words, **everything** our NRC does, in the order it happens.
For a shorter overview read [Understanding.md](Understanding.md) first.

Contents:

1. The idea in one minute
2. Words you will see
3. The numbers that control everything
4. The life of one sample (big picture)
5. Step by step: the life of a ray
6. Step by step: how the network learns
7. What the network sees (the 53 input numbers)
8. What the network predicts and how it is judged
9. Which file does what
10. Command line options
11. Reading the debug output
12. Things that are not done yet / known limits

---

## 1. The idea in one minute

A path tracer follows light paths for many bounces. That is slow and noisy.

NRC does this instead:

- Most paths are **stopped early**. At the stop point we ask a small neural network:
  "how much light leaves this point toward the camera?" and use its answer.
- A few paths (the **training paths**) are **not** stopped. They are traced for real.
  Their real result is used to **teach the network** what the right answer is.

So the network is a **learned memory of the light in the scene**.
It starts knowing nothing, and it learns while the image is rendering.

```
                    most paths                         few paths
                    (render paths)                     (training paths)

 camera ---> hit ---> hit ---> STOP, ask network       camera ---> hit ---> hit ---> keep going ---> real answer
                              |                                                      |
                              v                                                      v
                       network says "L = ..."                              teach the network
```

---

## 2. Words you will see

| Word | Meaning in simple terms |
|---|---|
| **Sample** (SPP) | One pass over the whole image. 128 SPP = 128 passes. |
| **Band** | A group of scanlines (rows) rendered together, to limit GPU memory. A sample is split into bands. |
| **Wavefront** | The GPU renders all rays at the same bounce number together, step by step (bounce 0 for all, then bounce 1 for all, ...). |
| **Vertex** | A point where a ray hit a surface. |
| **Depth** | The bounce number. Depth 0 is the first thing the camera sees. |
| **Throughput (beta)** | "How much light is still left after all bounces so far". Starts at 1, gets multiplied at every bounce. |
| **Query vertex** | The vertex where a path stops and the network is asked (or, for training paths, where the network's learning example starts). |
| **Area spread** | A number that tells us how blurry the light has become after bounces. It decides where the query vertex is (see step 5). |
| **Render path** | A normal path. After warmup it stops at the query vertex and uses the network. |
| **Training path** | A path that does NOT stop at the query vertex. It keeps going so we get a real answer to teach with. |
| **Suffix** | The part of a training path that comes AFTER the query vertex. It is the "extra tail" we trace for real. |
| **Bootstrap** | When the tail is cut short, we ask the network for "the rest" instead of tracing it. The network helps teach itself. |
| **RR-only path** | A training path whose tail is never cut. It ends only when Russian roulette (random death) kills it, or it escapes, so the answer is honest. 1 out of 16 training paths is this kind. |
| **Record** | One teaching example: inputs (what the point looks like) plus the target (the right answer). |
| **Target** | The correct radiance we want the network to learn. |
| **Warmup** | The first N samples. The network is trained but not used yet, because it knows nothing. |
| **Reflectance (R)** | How much light a surface bounces back. The network learns "light divided by R", and we multiply R back. |
| **Spectral** | pbrt traces 4 wavelengths per path (not RGB). The network input and output are about those 4 wavelengths. |

---

## 3. The numbers that control everything

| Setting | Value | Where | What it does |
|---|---|---|---|
| Warmup samples | 16 (change with `--nrc-warmup-samples`) | `options.h` | Network is only trained, not used, for this many samples. |
| Training budget | 65,536 records per training step | `kNRCTrainingBudget` | How many examples we teach with per sample. |
| Batches per step | 4 batches of 16,384 | `integrator.cpp` | The budget is split in 4 non-overlapping batches, one training step each. |
| Starting tile | 5 x 4 pixels | `integrator.cpp` | One training path per tile of 20 pixels. Later adapted. |
| Tile tolerance | records within a small percentage of the budget = leave it | `kNRCTileTolerance` | Prevents the tile size from jumping around. |
| Spread constant `c` | 0.01 | `kNRCSpreadC` in `surfscatter.cpp` | How blurry the light must be before we stop a path. |
| Max suffix length | 4 vertices | `kNRCMaxSuffixLen` in `workitems.h` | A normal training tail has at most 4 vertices. |
| Extra depth after warmup | +4 | `kNRCRROnlySuffixDepthCap` | Training paths are allowed to go this much deeper than `maxDepth` so there is room for the tail. |
| RR-only share | 1 of 16 training paths | `Hash(x,y,sample) & 15 == 0` in `camera.cpp` | Honest, unbiased examples. |
| Train every ... sample | every 4th sample after warmup (every sample during warmup) | `camera.cpp` | Training paths are only created on those samples. |
| Bright-record guarantee | on by default | `--nrc-brightness-oversampling` | Rare bright examples get reserved slots in each batch. |
| Network | Fully fused MLP, 5 hidden layers of 64 neurons, ReLU | `src/nrc/nrc_config.json` | The "brain". |
| Optimizer | Adam, learning rate 0.01 | `src/nrc/nrc_config.json` | How the brain learns. |

---

## 4. The life of one sample (big picture)

One sample = one pass over the whole image. The function is `WavefrontPathIntegrator::Render()` in `integrator.cpp`.

```
for each SAMPLE (0, 1, 2, ...):

    records collected so far = 0

    for each BAND of scanlines:
        A. reset the NRC buffers
        B. camera creates the rays                     (camera.cpp)
        C. for each bounce depth:                      (integrator.cpp + surfscatter.cpp)
              find hits, add emission,
              evaluate material, find query vertices,
              shoot light (shadow) rays, make next ray
        D. ask the network for stopped render paths    (NRCInferenceForRenderPaths)
        E. write the pixels                            (UpdateFilm)
        F. build the answers for training tails        (NRCTrainingSuffixFinish)
        G. put the finished examples in one big list   (NRCAccumulateTrainingRecords)

    train the network ONCE, using the list             (NRCTrainAccumulatedRecords)
    adapt the tile size                                (NRCUpdateTrainingTile)
```

Why train once per sample and not once per band? All bands of one sample should see the **same** network,
so the picture stays consistent.

After the last sample, an optional extra pass asks the network about every pixel's first hit
and writes that "what the network thinks" image to the file given by `--nrc-output`.

---

## 5. Step by step: the life of a ray

Follow one ray from the camera to the end.

### Step 1: The camera creates the ray (`camera.cpp`)

- pbrt creates a normal camera ray for the pixel.
- NRC decides if this ray is a **training path** or a **render path**:
  - The image is cut into small tiles (starting 5 x 4 pixels).
  - In each tile, one pixel is picked. The picked spot **moves every sample** (it uses a hash of the sample number), so over time all pixels get picked.
  - Training paths are only created if this is a training sample: all samples during warmup, then only every 4th sample.
- If it is a training path, it is also flagged as **RR-only** with a 1 in 16 chance.
- The camera position is saved as "the previous point" (`nrcPathPrevP`). We need it for the area spread.

```
Pixel A: normal           -> render path
Pixel B: picked by tile   -> training path
Pixel C: picked by tile   -> training path, and RR-only (1 in 16)
```

### Step 2: The ray flies and hits a surface (`integrator.cpp`)

- pbrt finds what the ray hits.
- If it hits a light, the emitted light is added to the pixel (with MIS weights, like normal pbrt).
- For training paths, the same emission is also written into the **suffix** buffers (see step 7), but **without** multiplying by throughput. The backward step does that later.

### Step 3: The material is evaluated (`surfscatter.cpp`)

At each hit, pbrt builds the BSDF (how the surface reflects). Next, NRC does its own bookkeeping.

### Step 4: Area spread: "is this the query vertex?" (`surfscatter.cpp`)

This follows the paper by Mueller et al. (2021).

Think about a flashlight. The farther the light travels and the more it scatters, the wider the lit spot.
After enough bounces, the light is so "blurry" that details no longer matter. The network can answer from there.

- **At the first hit (depth 0):** we measure `a0`, the size of the area one camera ray covers. `a0 = |x0 - x1|^2 / (4 pi |cos theta|)`.
- **At later hits:** we add up `sqrt(d^2 / (pdfPrev * |cos theta|))`, where `d` is the distance from the previous vertex and `pdfPrev` is the chance we had of choosing that direction.
- **The test:** if `(sum)^2 > c * a0` (with `c = 0.01`), this vertex is the **query vertex**.

A vertex is **also** the query vertex if:

- it is at `maxDepth - 1` (we must stop somewhere),
- the BSDF sample failed (the path cannot continue),
- Russian roulette killed the path here.

Not a query vertex yet? Then we save the current position and pdf for the next bounce's calculation, and the path continues normally.

### Step 5: At the query vertex, NRC writes down what the point looks like (`surfscatter.cpp`)

The 53 input numbers of the network are written (see section 7). Also stored:

- the reflectance `R` (4 numbers, one per wavelength),
- the luminance weights (4 numbers, used by the loss),
- `beta`, the throughput so far (render paths only), so we can scale the answer later.

### Step 6a: Render path (most pixels)

**Before warmup:** nothing special happens. The path just keeps going like in normal pbrt.

**After warmup:** the path is **stopped** right here (`nrcTerminateAndSubstitute`).

- No light sample (NEE) from this vertex.
- No next ray.
- It waits for the network.

```
render path:   camera ---> v0 ---> v1 ---> v2 (query vertex)  X  stop
                                              |
                                              +--> ask network later
```

### Step 6b: Training path (few pixels)

The path **keeps going for real**, but we start a new, independent bookkeeping called the **suffix**.

- The suffix starts with **throughput = 1** (a clean start).
- At every suffix vertex we store three things:
  - `inputs`: the 53 numbers of that vertex,
  - `local`: the light picked up at this vertex itself (emission + light samples/NEE),
  - `step`: how much of the next vertex's light makes it back to this one (`f * cos / pdf`, with the Russian roulette correction).
- The suffix has its **own** area spread tracking to decide where it ends.

The suffix ends in one of three ways:

| How it ends | What happens |
|---|---|
| **Natural end**: the ray escapes the scene, the BSDF sample fails, or Russian roulette kills it | There is nothing more to add. The tail is complete and real. |
| **Heuristic end**: the spread test says "blurry enough" | We stop and **ask the network** for the rest (bootstrap). Only after warmup. |
| **Cap**: it already has 4 vertices | Same as the heuristic end, a bootstrap. |

An **RR-only** training path never uses the heuristic or the cap. It keeps going until it ends naturally.
It does not store per-vertex steps. It just accumulates one single real answer for the query vertex.
This gives the network honest examples that do not depend on its own guesses.

```
training path: camera ---> v0 ---> v1 ---> [Q = query vertex] ---> s1 ---> s2 ---> s3 ---> (bootstrap: ask network)
                                              \______ suffix  ______/
```

### Step 7: Rays are finished, now the network is asked (`NRCInferenceForRenderPaths`)

Only after warmup.

1. Collect all stopped render paths (one batch).
2. Run the network (`Inference`). It returns `q` (4 numbers, one per wavelength).
3. For each stopped path add to the pixel:

```
L_pixel += beta * max(0, q) * R
```

- `q` is what the network says.
- `R` is the reflectance at the query vertex (we multiply it back, the network learned "light / R").
- `beta` is the throughput that got the path to that vertex.

This replaces everything the path would have collected after the query vertex.

### Step 8: The pixel is written (`UpdateFilm`)

The accumulated `L` is handed to pbrt's film, exactly like normal pbrt.
(Older versions also saved a training target here. That is **no longer** the case.)

### Step 9: Training tails get their answers (`NRCTrainingSuffixFinish`)

For every training suffix we now know `local[s]` and `step[s]` for each vertex `s`.

1. **Bootstrap (only after warmup):** for tails cut short, the network is asked about the last vertex. The result, times R, is `Lnext` for the end of the chain. For natural ends `Lnext = 0`.
2. **Walk backward** from the last vertex to the query vertex:

```
Ls = local[s] + step[s] * L_{s+1}
```

In words: *"the light leaving this vertex = the light it picks up right here + the part of the next vertex's light that survives the bounce."*

Example with 3 vertices:

```
L2 = local2 + step2 * Lnext          (Lnext = 0 or bootstrap)
L1 = local1 + step1 * L2
L0 = local0 + step0 * L1             <-- L0 is the target for the query vertex
```

Every vertex gets its own target: L0, L1 and L2 become three teaching examples from one path.

For RR-only suffixes there is nothing to walk back. The one accumulated value is already the target.

**Why no bootstrap before warmup?** An untrained network gives random, often huge numbers.
Putting these into targets would teach the network to answer even bigger numbers, which can explode to NaN.
So before warmup, the missing continuation is treated as zero.

### Step 10: Examples go into one big list (`NRCAccumulateTrainingRecords`)

On the GPU, all valid examples from this band are copied into one tight list (`nrcCompact*` buffers),
placed after what earlier bands already stored.
Each example holds: the 53 inputs, the target `Ls`, and 8 extra numbers (aux): `[R0..R3 | w0..w3]`
(`R` floored at 0.001, `w` = luminance weights).

When all bands are done we have **all examples of this sample** in one list.

---

## 6. Step by step: how the network learns (`NRCTrainAccumulatedRecords`)

This runs **once per sample**, after all bands.

1. **How many examples?** `available` = all collected. We use at most `65,536` (4 batches x 16,384).
2. **Choose which ones.** Two modes:
   - **Brightness oversampling ON (default):** bright examples (weighted target brightness at least 0.1) are rare but important
     (lights, caustics, reflections). They get a **guaranteed share** of slots: `max(brightCount, used / 800)`, repeated if there are few of them.
     The rest of the slots are filled by a random shuffle.
   - **OFF** (`--nrc-brightness-oversampling=false`): a plain random shuffle.
3. **Gather on the GPU** into 4 contiguous batches (no data is copied back to the CPU).
4. **Train.** For each batch, one `TrainN` step. So there are 4 small gradient steps per training sample.
   The loss is `SpectralRelativeL2` (section 8).
5. **Update the state.**
   - `nrcSampleCounter++`
   - `nrcWarmedUp = (nrcSampleCounter >= warmup samples)`
6. **Adapt the tile** (`NRCUpdateTrainingTile`). The goal is about 65,536 examples per sample:
   - Compare the number of examples to the goal.
   - Too few: make the tile smaller (more training paths). Too many: make it bigger.
   - Changes are limited (at most halving or doubling the area), the area is kept between 4 and 128 pixels, and the shape is kept close to 2:1.
   - It is only changed between samples, never between bands.

On samples where no training paths are made (3 of every 4 after warmup), there are 0 examples,
so no training happens, the counter still goes up, and the tile is left unchanged.

### Timeline of the network's life

```
sample:     0 ........ 15 | 16 ... 
state:      WARMUP         | WARMED UP
render paths: normal pbrt  | stop at the query vertex, ask network
training:   every sample   | every 4th sample
suffix end: natural only   | natural / heuristic / cap (bootstrap)
bootstrap:  no             | yes
maxDepth:   normal         | +4 for training paths (room for the tail)
```

---

## 7. What the network sees: the 53 input numbers

Written in `surfscatter.cpp`. The count is `kNRCInputDims = 49 + NSpectrumSamples = 53`.
Some are passed through as they are, some are expanded by the network's encoder (see `nrc_config.json`).

| Index | Count | What | Encoding |
|---|---|---|---|
| 0 to 35 | 36 | Position, scaled into the scene box (0 to 1), as `sin(2^d * x)` for 12 frequencies and 3 axes (x, y, z) | Identity (frequencies done by us) |
| 36, 37 | 2 | Outgoing direction `wo` as (theta, phi) scaled to 0..1 | OneBlob (4 bins) |
| 38, 39 | 2 | Shading normal as (theta, phi) scaled to 0..1 | OneBlob (4 bins) |
| 40 | 1 | Roughness, transformed `1 - exp(-roughness)` | OneBlob (4 bins) |
| 41 to 43 | 3 | Diffuse reflectance alpha, as RGB, clamped to 0..1 | Identity |
| 44 to 46 | 3 | Specular reflectance beta, as RGB, clamped to 0..1 | Identity |
| 47 to 50 | 4 | The 4 sampled wavelengths of this path, scaled to 0..1 | Identity |
| 51, 52 | 2 | Padding, always 1 | Identity |

Why the wavelengths? The output is "light at these 4 wavelengths". Without them the output would mean nothing.

How alpha and beta are found (the `R = alpha + beta` split):

- Diffuse surface: `alpha = R`, `beta = 0`.
- Dielectric (glass-like): `beta` = reflectance estimated with 16 fixed samples, reflection only.
- Conductor (metal): `beta` = the normal reflectance estimate (`bsdf.rho`), `alpha = 0`.
- Anything else: the combined estimate is used as `alpha`, `beta = 0`.

---

## 8. What the network predicts and how it is judged

**Output:** 4 numbers `q` (one per wavelength).

**Real radiance:** `L_hat = R * q`.
We "factor out" the reflectance so that the network does not have to relearn every colour of every material.
It only learns how much light *arrives*. The surface colour is multiplied back.

**Loss (`SpectralRelativeL2`, in tiny-cuda-nn):**

- Compares `R * q` with the target `Ls`.
- The error is divided by `(brightness of the prediction)^2 + 0.01`, where brightness uses the CIE luminance weights.
- So the loss is about **relative** error: a dark area off by 0.01 matters as much as a bright area off by 1.
  Without this, the few very bright records would drown the rest.

**Where `R * q` is used (always the same way):**

| Place | Use |
|---|---|
| Render substitution | `L += beta * max(0, q) * R` |
| Bootstrap | `Lnext = max(0, q) * max(R, 0.001)` |
| Loss | `R * q` vs `Ls` |

---

## 9. Which file does what

| File | Role |
|---|---|
| `src/pbrt/wavefront/camera.cpp` | Creates rays. Chooses training paths (tile, sample, 1/16 RR-only). Saves the camera position. |
| `src/pbrt/wavefront/surfscatter.cpp` | The heart. At every hit: area spread, query vertex, writes the 53 inputs, stops render paths, tracks the training suffix (`local`, `step`, `inputs`, `R`, weights). |
| `src/pbrt/wavefront/integrator.cpp` | Coordinator. Allocates buffers. The sample loop. Emission added to suffix. Light samples (NEE) added to suffix. `NRCInferenceForRenderPaths`, `NRCTrainingSuffixFinish`, `NRCAccumulateTrainingRecords`, `NRCTrainAccumulatedRecords`, `NRCUpdateTrainingTile`, debug output. |
| `src/pbrt/wavefront/integrator.h` | Declares all `nrc*` buffers and functions. |
| `src/pbrt/wavefront/workitems.h` | Constants such as `kNRCMaxSuffixLen`, `kNRCRROnlySuffixDepthCap`. |
| `src/pbrt/wavefront/film.cpp` | Writes pixels. It no longer builds training targets. |
| `src/pbrt/options.h`, `src/pbrt/cmd/pbrt.cpp` | The command line options. |
| `src/nrc/nrc.cu`, `src/nrc/nrc.h` | Wrapper around tiny-cuda-nn: `TrainN`, `InferenceN`, `RoundUpBatch`. |
| `src/nrc/nrc_config.json` | Network, encoder, loss and optimizer settings. |
| `external/tiny-cuda-nn/.../spectral_relative_l2.h` | Our loss. |

Everything NRC is behind `#ifdef PBRT_BUILD_NRC` (build with `-DPBRT_BUILD_NRC=ON`) and is only active with `--enable-nrc` on the GPU.

---

## 10. Command line options

| Option | Meaning |
|---|---|
| `--enable-nrc` | Turn NRC on. |
| `--nrc-warmup-samples N` | Number of warmup samples. Default 16. |
| `--nrc-config file.json` | Network configuration file. |
| `--nrc-output file.exr` | Write the "network only" picture here after rendering. |
| `--nrc-debug` | Print diagnostics to stderr (see section 11). |
| `--nrc-brightness-oversampling=false` | Turn off the bright-record guarantee. |
| `--nrc-train-steps N` | Exists in the options but is **not used** by the code right now. |

Important: true/false options need the `=` form. `--nrc-brightness-oversampling false` does **not** work. Use `--nrc-brightness-oversampling=false`.

Example:

```
./build/pbrt scene.pbrt --gpu --spp 128 --enable-nrc \
    --nrc-config src/nrc/nrc_config.json --nrc-output predicted.exr \
    --nrc-debug 2> nrc_debug.log
```

---

## 11. Reading the debug output (`--nrc-debug`)

Everything goes to stderr, so use `2> file` to save it.

| Line starts with | What it tells you |
|---|---|
| `NRC TRAIN ACCURACY BY TARGET BRIGHTNESS` | A table: for each brightness range, how many records, and `pred/target`. Close to 1.0 is good. Below 1 means the network under-predicts (too dark). |
| `NRC TILE` | The tile size changed: how many records were made, the old and the new tile. |
| `OUTPUT BEFORE TRAIN` / `OUTPUT AFTER TRAIN` | Checks for NaN/Inf and the range of the network outputs. |
| `NRC max-target trace` | Only appears if one target is huge (above 1000). Prints the full backward walk to find the culprit. |
| `prediction` / `target` stats | Magnitude of what the network predicts and of the targets (every 32 samples). |
| Query-depth histogram | At what bounce depth the render paths were stopped. |
| Suffix NEE generated/visible | How many light samples of training tails were made and how many were not blocked. |

---

## 12. Known limits and ideas

Done (these used to be on the TODO list):

- Training suffix with its own area spread and cap.
- 1/16 unbiased RR-only suffixes.
- Backward target calculation through the suffix.
- Reflectance factorization (`R = alpha + beta`, `R * q`).
- Spectral (4 wavelength) input and output.
- Tile adaptation to a constant record budget.
- GPU-side compaction and gathering of records.

Things to know:

- The old "first hit record" (`nrcValid`, `nrcTargets`) is no longer used for training. Only suffix records train the network.
- `--nrc-train-steps` is not read anywhere. The number of steps is fixed by the 4 batches.
- Training only on every 4th sample after warmup, and the bright-record share of `used / 800`, are current experiments. They trade a bit of learning speed for lower cost per sample.
- Materials other than diffuse, dielectric and conductor use the generic reflectance estimate (no alpha/beta split).
- Paper-fidelity details that may still differ: network size, no weight averaging (EMA), our own encoding choices.
