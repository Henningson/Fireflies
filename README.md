![Fireflies](https://github.com/Henningson/Fireflies/assets/27073509/36254690-b42a-4604-849f-ebfa4ffa69c6)

**Fireflies** is a wrapper for the <a href="https://mitsuba.readthedocs.io/en/latest/">Mitsuba Renderer</a> and allows for rapid prototyping and generation of physically-based renderings and simulation data in a differentiable manner.
It can be used for example, to easily generate highly realistic medical imaging data for medical machine learning tasks.
I originally created it to research if the task of finding an optimal point-based laser pattern for structured light laryngoscopy can be reformulated as a gradient-based optimization problem. 
That is also why you'll find a lot of Single-Shot Structured Light specific stuff in the code.

This repository accompanies the paper **Fireflies: Domain-specific Structured Light
Optimization for Medical 3D Reconstruction** published at MICCAI'24.
The code for the paper can be found in the paper-branch.
This is a joint work of the <a href="https://www.lgdv.tf.fau.de/">Chair of Visual Computing</a> of the Friedrich-Alexander University of Erlangen-Nuremberg and the <a href="https://www.hno-klinik.uk-erlangen.de/phoniatrie/">Phoniatric Division</a> of the University Hospital Erlangen. 

# Main features
- **Easy torch-like and pythonic scene randomization description.** This library is made to be easily usable for everyone who regularly uses python and pytorch. We implement train() and eval() functionality from the get go.
- **Integratable into online deep-learning and machine learning tasks** due to the differentiability of the mitsuba renderer w.r.t. the scene parameters.
- **Simple animation description**. Have a look into the examples.

# Installation
Fireflies is hosted on Pypi. You can either install it via
```
pip install fireflies
```

or by cloning this repository. Make sure to create a conda environment first.
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
They span from defining a simple scene to training neural networks and optimizing point-based structured light pattern.
Ideally, you work through them one by one. They consist of:

1. **Hello World** - How to wrap fireflies around your Mitsuba scene.
2. **General Transformations** - Showcasing different affine transformations.
3. **Parent Child** - Defining hierarchical relationships for objects in the scene.
4. **Material Randomization** - How to randomize material parameters
5. **Light Randomization** - How to randomize lights
6. **Animation** - Apply deformations either by scripting, or by loading meshes from a folder.
7. **Optimization** - Using fireflies for optimization of different parameters.
8. **Gradient Accumulation** - Using Gradient Accumulation for smoother update steps during optimization.
9. **Laser Pattern Creation** - How to define and create laser pattern highlighted in the paper. 
10. **Laser Pattern Optimization** - Laser pattern optimization to reduce ambiguities in correspondence estimation.
11. **Domain Specific Pattern Optimization** - Optimize a laser pattern that minimizes a specific target function. For paper readers, this is the Gaussian optimization task. More paper specific code can be found in the **paper** branch.


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

## PSA
Since I am now in my last year of my PhD, I won't be really able to further work on this library for the time being.
Please start pull requests for features, Add-Ons, Bug-fixes, etc. I'd be very happy about any help. :)

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
@InProceedings{HenningsonFireflies,
  author="TBD",
  title="TBD",
  booktitle="TBD",
  year="2023",
  pages="?",
  isbn="?"
}
```
