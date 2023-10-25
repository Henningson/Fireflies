# Fireflies
**Fireflies** is a tool for the domain-specific optimization and generation of laser-based structured light pattern.
This repository accompanies the paper **Fireflies: Laser-based Single Shot Structured Light a la Carte**.
This is a joint work of the <a href="https://www.lgdv.tf.fau.de/">Chair of Visual Computing</a> of the Friedrich-Alexander University of Erlangen-Nuremberg and the <a href="https://www.hno-klinik.uk-erlangen.de/phoniatrie/">Phoniatric Division</a> of the University Hospital Erlangen. 

https://github.com/Henningson/Fireflies/assets/27073509/911e57d3-3aab-418d-ad02-e8721cd36785

# Building your own Scene
First, make sure you have the <a href="https://github.com/mitsuba-renderer/mitsuba-blender">Mitsuba Blender Add-On</a> installed.
The Firefly Blender Add-On can be found in the ```blender``` folder.


# Regarding the Epipolar Constraint regularization
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
A simple example with an $L1$ loss is given here:

https://github.com/Henningson/DSLPO/assets/27073509/18e3d63c-3b3e-4ca0-92d5-a2c28b9ebb44


This can easily be repurposed for other primitive types.
Here, another example with points:

https://github.com/Henningson/DSLPO/assets/27073509/aedea7b6-8e5e-40c1-85ca-31760bd60a70
