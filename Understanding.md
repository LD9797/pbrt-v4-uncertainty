## camera.cpp 

"Birth of the path"

```
camera.cpp
    |
    +-- Create normal PBRT camera path
    |
    +-- NRC: mark whether path is a training path
    |
    +-- NRC: save x0 (camera position) for area-spread calculation
```

## film.cpp 

It saves the RGB target for paths selected for NRC training.

```
training path finishes
    ↓
get its final radiance
    ↓
build RGB target
    ↓
store in nrcTargets
```

## integrator.cpp

The main coordinator of the whole wavefront renderer.

```
integrator.cpp
    ↓
create NRC network + buffers
    ↓
for each sample / scanline
    ↓
camera.cpp creates paths
    ↓
PBRT traces them
    ↓
surfscatter.cpp may find NRC query vertices
    ↓
NRC inference substitutes radiance for terminated render paths
    ↓
film.cpp saves training targets
    ↓
train NRC
```

In the constructor, it allocates things like `nrcInputs`, `nrcTargets`, `nrcTrainingPath`, `nrcReachedQueryVertex`, `area-spread buffers`, `inference output buffers`, etc., and creates the `NeuralRadianceCache`.


Before each new batch of paths, it calls:
```C++
NRCResetSampleBuffers();
```
so the NRC state starts clean.

After PBRT finishes tracing the paths, it calls:
```C++
NRCInferenceForRenderPaths();
```
This is where terminated normal rendering paths get the NN prediction inserted into their accumulated radiance.

Then:
```C++
UpdateFilm();
```

saves the image sample and training RGB targets.

Finally:

```C++
NRCTrainAndInferStep();
```

takes those collected training examples and trains the network.

## surfscatter.cpp 

`surfscatter.cpp` is where the path interacts with a surface and decides what happens next. 


Normal PBRT job:

```
ray hits surface
    ↓
evaluate material / BSDF
    ↓
sample next direction
    ↓
create next bounce ray
    ↓
continue path
```

NRC changes: 

- Calculate and accumulate Müller's area-spread heuristic.
- When the threshold is reached, mark:

```C++ 
nrcReachedQueryVertex[pixelIndex] = 1;
```

- Save the state at that point, such as:

```C++ 
nrcSnapshotL
nrcSnapshotBeta
```
- Build and save the neural-network input features for that surface into: 

```C++ 
nrcInputs
``` 
including position, direction, normal, roughness, diffuse/specular reflectance, etc.

- If it's a training path, let it continue tracing so we can eventually obtain a target.
- If it's a normal rendering path, after warmup:
```C++
nrcTerminateAndSubstitute = true;
```
so it does not create the next bounce ray; later integrator.cpp replaces the remaining path with the NRC prediction.
- While the path continues, it stores the previous position and BSDF PDF needed for the next area-spread calculation. 


# Resume: 

```
camera.cpp
→ choose sparse training paths + initialize path tracking

surfscatter.cpp
→ track area spread
→ choose query vertex
→ capture NN inputs
→ terminate normal paths / continue training paths

integrator.cpp
→ coordinate buffers, inference and training

film.cpp
→ construct/save the RGB training target
```

# TODO: 

- Target definition is still unresolved. Your current Lraw - Lsnapshot version is an experiment, not the final NRC target.
- We haven't implemented Müller's proper training suffix behavior yet.
- We haven't implemented the 1/16 unbiased RR-only training suffixes.
- We haven't implemented the full self-training / backward target propagation through the training suffix.
- A few paper-fidelity details remain such as architecture/EMA/reflectance factorization.