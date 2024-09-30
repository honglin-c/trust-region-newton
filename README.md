# Trust-Region Eigenvalue Filtering for Projected Newton

### [Project Page](https://www.cs.columbia.edu/cg/trust-region/)  | [Paper](https://www.cs.columbia.edu/cg/trust-region/paper.pdf)

<img src="https://github.com/honglin-c/trust-region/blob/main/.github/images/teaser.png" width="800">

Official implementation for the paper:
> **[Trust-Region Eigenvalue Filtering for Projected Newton](https://www.cs.columbia.edu/cg/trust-region/)**  
> [Honglin Chen](https://www.cs.columbia.edu/~honglinchen/)<sup>1</sup>, 
[Hsueh-Ti Derek Liu](https://www.dgp.toronto.edu/~hsuehtil/)<sup>3,</sup><sup>4</sup>, 
[Alec Jacobson](https://www.cs.toronto.edu/~jacobson/)<sup>2,</sup><sup>6</sup>,
[David I.W. Levin](http://www.cs.toronto.edu/~diwlevin/)<sup>2,</sup><sup>5</sup>, 
[Changxi Zheng](http://www.cs.columbia.edu/~cxz/)<sup>1</sup>
<br>
> <sup>1</sup>Columbia University, 
<sup>2</sup>University of Toronto,  
<sup>3</sup>Roblox Research, 
<sup>4</sup>University of British Columbia, 
<br> 
&nbsp; &nbsp; <sup>5</sup>NVIDIA,
<sup>6</sup>Adobe Research
<br>
> SIGGRAPH Asia 2024 (Conference Track)


## Installation
To build the code, please run
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

## Examples

To run the code with our absolute eigenvalue projection strategy, please run, e.g.,
```
sh ../scripts/micky_teaser/abs.sh
```
We provide several examples in `scripts/`. 

### Comparisons
To run the same example with the eigenvalue clamping strategy, run `clamp.sh` under the same folder (or add `--clamp` option after the command in `abs.sh`):
```
sh ../scripts/frog.sh
```
The default eigenvalue clamping algorithm uses 0 as the clamping threshold. To use a different clamping threshold (e.g., a small positive number), add `--epsilon [threshold]` option after the command. 

## Experiments

We provide the python scripts to run the experiments in our paper.
To run the experiment of each figure, please run, e.g.,
```
python ../experiments/teaser_frog.py
```

The python script will iteratively run the example with different eigenvalue filtering strategies, deformation magnitudes and Poisson's ratios. (You'll need to close the libigl window before proceeding to the next configuration. )

## Optional arguments

For more options, please see
```
./example --help
```

<details>
<summary>
<h3>What do we modify in TinyAD to add our absolute eigenvalue projection? </h3>
</summary>

As a research prototype, we choose to make minimal modifications in TinyAD when adding our new projection method. 
We clone [TinyAD](https://github.com/patr-schm/TinyAD/blob/29417031c185b6dc27b6d4b684550d844459b735D) to the project folder,
and comment out and change [lines 71-75](https://github.com/patr-schm/TinyAD/blob/29417031c185b6dc27b6d4b684550d844459b735/include/TinyAD/Utils/HessianProjection.hh#L71-L75) in `TinyAD/include/TinyAD/Utils/HessianProjection.hh` to:
```
  if (_eigenvalue_eps < 0) {
      // project to absolute value if the eigenvalue threshold is set to be less than 0
      if (D(i, i) < 0)
      {
          D(i, i) = -D(i, i);
          all_positive = false;
      }
  }
  else {
      // project to epsilon otherwise
      if (D(i, i) < _eigenvalue_eps)
      {
          D(i, i) = _eigenvalue_eps;
          all_positive = false;
      }
  }
```
Thus we simply use `eps < 0` as a flag for absolute eigenvalue projection.

</details>

<details>
<summary>
<h3>Wait, for this specific example, the adaptive method is slightly slower than absolute eigenvalue projection / eigenvalue clamping? </h3>
</summary>

</details>


## Citation
```
@inproceedings{chen2024trust_region,
      title={Trust-Region Eigenvalue Filtering for Projected Newton},
      author={Honglin Chen and Hsueh-Ti Derek Liu and Alec Jacobson and David I.W. Levin and Changxi Zheng},
      booktitle = {ACM SIGGRAPH Asia 2024 Conference Proceedings},
      year = {2024}
  }
```