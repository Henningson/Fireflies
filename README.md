![Fireflies](https://github.com/Henningson/Fireflies/assets/27073509/36254690-b42a-4604-849f-ebfa4ffa69c6)

**Fireflies** is a wrapper for the <a href="https://mitsuba.readthedocs.io/en/latest/">Mitsuba Renderer</a> and allows for rapid prototyping and generation of physically-based renderings and simulation data in a differentiable manner.
It can be used for example, to easily generate highly realistic medical imaging data for medical machine learning tasks or (its intended use) test the reconstruction capabilities of Structured Light projection systems in simulated environments.
I originally created it to research if the task of finding an optimal point-based laser pattern for structured light laryngoscopy can be reformulated as a gradient-based optimization problem. 

This code accompanies the paper **Fireflies: Photorealistic Simulation and Optimization of Structured Light Endoscopy** accepted at **SASHIMI 2024**. 🎊


# Main features
- **Easy torch-like and pythonic scene randomization description.** This library is made to be easily usable for everyone who regularly uses python and pytorch. We implement train() and eval() functionality from the get go.
- **Integratable into online deep-learning and machine learning tasks** due to the differentiability of the mitsuba renderer w.r.t. the scene parameters.
- **Simple animation description**. Have a look into the examples.
- **Single Shot Structured Light specific**. You can easily test different projection pattern and reconstruction algorithms on randomized scenes, giving a good estimation of the quality and viability of patterns/systems/algorithms.

# Installation
Make sure to create a conda environment first.
I tested fireflies on Python 3.10, it should however work with every Python version that is also supported by Mitsuba and Pytorch.
I'm working on adding Fireflies to PyPi in the future.
First install the necessary dependencies:
```
pip install pywavefront geomdl
pip install torch
pip install mitsuba
```
To run the examples, you also need OpenCV:
```
pip install opencv-python
```
Finally, you can install Fireflies via:
```
git clone https://github.com/Henningson/Fireflies.git
cd Fireflies
pip install .
```

![Datasets](https://github.com/Henningson/Fireflies/assets/27073509/9c617876-356a-420d-8632-cf4c286d6778)
# Usage
```
import mitsuba as mi
import fireflies as ff

mi_scene = mi.scene(path)
mi_params = mi.traverse(mi_scene)
ff_scene = ff.scene(mi_params)

mesh = ff_scene.mesh_at(0)
mesh.rotate_z(-3.141, 3.141)

ff_scene.eval()
#ff_scene.train() generates uniformly sampled results on the right
for i in range(0, 20):
    ff_scene.randomize()
    mi.render(mi_scene)
```

<p align="center">
<img src="https://github.com/Henningson/Fireflies/assets/27073509/78e1af22-d526-4130-adc6-d3b30c2cc4d9"/>
<img src="https://github.com/Henningson/Fireflies/assets/27073509/882f30b8-8254-493a-9c81-2be702c83326"/>
</p>

# Examples
A bunch of different examples can be found in the examples folder.
Papercode can be found in the **Paper** branch.

# Building and loading your own scene
You can easily generate a scene using Blender.
To export a scene in Mitsubas required .xml format, you first need to install the <a href="https://github.com/mitsuba-renderer/mitsuba-blender">Mitsuba Blender Add-On</a>.
You can then export it under the File -> Export Tab.  
Make sure to tick the ✅ export ids Checkbox, as fireflies infers the object type by checking for name qualifiers with specific keys, e.g.: "mesh", "brdf", etc.

# Render Gallery
<p align="center">
<img src="https://github.com/Henningson/Fireflies/assets/27073509/dce49ad1-1d22-45b3-a544-2e1fbcd7b30c" height="150"/>
<img src="https://github.com/Henningson/Fireflies/assets/27073509/f92fad5f-0913-40c8-947f-fa260f19c26e" height="150"/>
<img src="https://github.com/Henningson/Fireflies/assets/27073509/429aa015-9987-4559-8776-b819f32ff81a" height="150"/>
<img src="https://github.com/Henningson/Fireflies/assets/27073509/68922274-344b-42f0-81f5-b65693e11006" height="150"/>
</p>
These are some renderings that were created during my work on the aforementioned paper.
From left to right: Reconstructed in-vivo colon, the flame shapemodel, phonating human vocal folds with a point-based structured light pattern.

## More Discussion about the Paper
Can be found in the **README** of the **paper** branch.

## Why did you call this Fireflies?
Because optimizing a point-based laser pattern looks like fireflies that jet around. :)  
<p align="center">
<img src="https://github.com/Henningson/Fireflies/assets/27073509/220217db-2a47-4eb2-869f-e39789922a70"/>
</p>


## Acknowledgements
A big thank you to Wenzel Jakob and team for their wonderful work on the Mitsuba renderer.
You should definitely check out their work: <a href="https://www.mitsuba-renderer.org/">Mitsuba Homepage</a>, <a href="https://github.com/mitsuba-renderer/mitsuba3">Mitsuba Github</a>.

Furthermore, this work was supported by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under grant STA662/6-1, Project-ID 448240908 and (partly) funded by the DFG – SFB 1483 – Project-ID 442419336, EmpkinS.


<p align="center">
<img src="https://github.com/Henningson/Vocal3D/blob/main/images/lgdv_small.png?raw=true" height="70"/> 
<img src="https://raw.githubusercontent.com/Henningson/Vocal3D/ac622e36b8a8e7b57a7594f1d12a4f34c81450f4/images/Uniklinikum-Erlangen.svg" height="70"/>
</p>

## Citation
Please cite this, if this work helps you with your research:
```
@InProceedings{10.1007/978-3-031-73281-2_10,
author="Henningson, Jann-Ole and Veltrup, Reinhard and Semmler, Marion and D{\"o}llinger, Michael and Stamminger, Marc",
title="Fireflies: Photorealistic Simulation and Optimization of Structured Light Endoscopy",
booktitle="Simulation and Synthesis in Medical Imaging",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="102--112",
isbn="978-3-031-73281-2"
}
```
