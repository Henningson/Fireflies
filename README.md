![Fireflies](https://github.com/Henningson/Fireflies/assets/27073509/36254690-b42a-4604-849f-ebfa4ffa69c6)

**Fireflies** is a wrapper for the <a href="https://mitsuba.readthedocs.io/en/latest/">Mitsuba Renderer</a> and allows for rapid prototyping and generation of physically-based renderings and simulation data in a differentiable manner.
It can be used for example, to easily generate highly realistic medical imaging data for medical machine learning tasks.
I originally created it to research if the task of finding an optimal point-based laser pattern for structured light laryngoscopy can be reformulated as a gradient-based optimization problem. 
That is also why you'll find a lot of Single-Shot Structured Light specific stuff in the code.

This repository accompanies the paper **Fireflies: Domain-specific Structured Light
Optimization for Medical 3D Reconstruction** published at MICCAI'24.
The code for the paper can be found in the paper-branch.
This is a joint work of the <a href="https://www.lgdv.tf.fau.de/">Chair of Visual Computing</a> of the Friedrich-Alexander University of Erlangen-Nuremberg and the <a href="https://www.hno-klinik.uk-erlangen.de/phoniatrie/">Phoniatric Division</a> of the University Hospital Erlangen. 


# Installation
```
conda env create -n Fireflies python=3.10

git clone this

install stuff
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
mesh.rotate_z(-math.pi, math.pi)

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

# Render Gallery
<p align="center">
<img src="https://github.com/Henningson/Fireflies/assets/27073509/dce49ad1-1d22-45b3-a544-2e1fbcd7b30c" height="150"/>
<img src="https://github.com/Henningson/Fireflies/assets/27073509/f92fad5f-0913-40c8-947f-fa260f19c26e" height="150"/>
<img src="https://github.com/Henningson/Fireflies/assets/27073509/429aa015-9987-4559-8776-b819f32ff81a" height="150"/>
<img src="https://github.com/Henningson/Fireflies/assets/27073509/68922274-344b-42f0-81f5-b65693e11006" height="150"/>
</p>

# More Discussion about the Paper
I think that the eight pages given in the MICCAI format is not enough to properly discuss everything. So here's some further discussion, thoughts and limitations.

## Why do we need scene-specific point pattern?
The point pattern used in single-shot structured light can generally be regarded as a sampling strategy.
However, different sampling strategies allow for different observations, for example, here, a general image with different sampling strategies is shown.
First, a gradient-based sampling, next a blue noise sampling, and lastly a vanilla random sampling.
Which of these point pattern lets you decipher the image the best?
<p align="center">
<img src="https://github.com/Henningson/Fireflies/assets/27073509/f8b09d19-98a5-4736-90a3-770b63ab0666" height="150"/>
</p>

When it comes to laser based single shot structured light, this is a good case for how the distribution of the laser pattern directly influences the reconstructability of the scene.
However, scenes are unlike images variable in its composition, i.e. objects may move, rotate or deform, or the camera may move, lighting changes, etc.  
So finding a point pattern that is scene specific and optimal is a highly ill-posed problem.
Thus, we often see structured light pattern, that are
 - easy to produce e.g. single line laser or symmetric point pattern,
 - or very general in its applicability.

However, if we think about specific scenes, we surely can find a pattern that increases the accuracy of our reconstruction.
And because its so nice, here's the actual optimization of the point pattern as a video:  

https://github.com/Henningson/Fireflies/assets/27073509/0afc5a79-d0ac-485a-a1f1-75451d0e4ed9
## Regarding the Epipolar Constraint regularization in the Paper
Our goal is to optimize a pattern that a) is purposely designed for the scene and b) does not contain any ambiguities.
We can achieve this, by making sure that the epipolar lines inside the operating range of the structured light projector do not overlap.
Naturally, this would lead to a **very** sparse pattern.
Thus, we relax this constraint and allow for intersections, but try to minimize these inside the optimization.
However, counting the number of intersections is a discrete problem and is not useful for a gradient-based optimization approach.
Thus, we need to find a regularization that parametrizes intersections in a continuous space.

We achieve this, by reformulating the following observation:
Let $A$ be a set of lines, and $I$ their rasterized representation in a $(N, H, W)$ tensor, where each $I_{(i)}$ contains ones in each cell the line crosses through and zeros elsewhere.
Then, lines do not cross if the *softor* representation (from "Differentiable Drawing and Sketching", Mihai and Hare) is equal to the sum over the line dimension.  

More rigorous for the discrete case:
```math
\forall \left(a_{x},b_{x}\right),\left(a_{y},b_{y}\right) \in A \nexists s,t \in [0, 1]: a_{x}+sb_{x}=a_{y}+st_{y}  
\Leftrightarrow  
1 - \left(\prod_{i=0}^n 1 - I_{(i)} \right) = \sum_{i=0}^n I_{(i)}
```

Now, we can repurpose this formulation in a continuous case, where $I^G_{(i)}$ is a differentiable (gaussian) representation of lines.
```math
\forall \left(a_{x},b_{x}\right),\left(a_{y},b_{y}\right) \in A \quad \nexists s,t \in [0, 1]: a_{x}+sb_{x}=a_{y}+st_{y}  
\Leftrightarrow  
\lim_{\sigma \rightarrow 0} \left(\sum_{i=0}^n I^G_{(i)} - \left(1 - \left(\prod_{i=0}^n 1 - I^G_{(i)} \right)\right)\right) = 0
```
This allows us to easily optimize via
```math
L\left(1 - \left(\prod_{i=0}^n 1 - I^G_{(i)} \right), \sum_{i=0}^n I^G_{(i)}\right)
```
where $L$ may be any arbitrary loss function.
This optimizable measure can give very crude information about the "ambiguityness of a chosen pattern".
If the measure is 0 then any laser point that is found in image space can be directly traced back towards the creating laser.

## Limitations
Right now the differentiable rasterization code needs a lot of VRAM. 


## PSA
Since I am now in my last year of my PhD, I won't be really able to further work on this library for the time being.
Please start pull requests for features, Add-Ons, Bug-fixes, etc. I'd be very happy about any help. :)

# Citations
TODO

