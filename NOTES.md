Generar imagen PBRT corriente GPU: 

 ./pbrt --gpu --spp 30 ../../pbrt-v4-scenes/barcelona-pavilion/pavilion-day.pbrt 

Generar imagen CPU: 

 ./pbrt ../../pbrt-v4-scenes/barcelona-pavilion/pavilion-day.pbrt 

 ----

Notes:

 - Position is being normalized into [0, 1] using scene bounds while the paper doesn't describe scene-bounds normalization. 
 - Specular reflectance is interpreted as Fresner F0. Uknown if that's exactly what Muller's implementation computes. 


 