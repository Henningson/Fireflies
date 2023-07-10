# Firefly
Generation and optimization of domain-specific laser pattern for single-shot structured light 3D reconstruction tasks.



# TODO
## Meshes
✅ Randomizable Vertices  
✅ Export Translation, Rotation, Scale Constraints from Blender  
✅ Import Translation, Rotation, Scale Constraints in Firefly  
✅ Apply Random Transformations  
✅ Sequential Randomization  
❌ Randomize Position along Curves/Splines  

## Camera
✅ Randomizable Position, Rotation  
✅ Export Translation, Position Constraints  
✅ Import and apply randomized translation, position  
✅ Randomize Position along Curves/Splines 

## Projector/Laser
✅ Export from Blender  
✅ Import in Firefly  
✅ Apply Cameras Transformation  

## Relative Objects
✅ Export  
✅ Loading in Firefly  
✅ Including Constraints into local transformations

## Initialization
✅ Randomized Depth Maps  
✅ Initialization of laser pattern  
❌ Weighted segmentation maps  
❌ Take epipolar lines into account  
❌ Laser Pattern FOV constraints  
❌ Weighted object significance  

## Optimization
❌ Averaged Gradient Step  
❌ UNet based Depthmap inpainting  
❌ Add noise  

## Experiments
❌ Semi Global Matching?  
❌ Evaluate Depth Map with and without Laserpattern  
❌ Improved ICP through optimized laser pattern  
❌ General correspondence estimation  
❌ Optimized correspondence estimation  

## Utils
❌ Change xml read to yaml -> dict read