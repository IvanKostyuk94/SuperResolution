# cosmoSR

A python package to train a neural network to super-resolve gridded cosmological simulations. <p>

```diff
- Note: This package is currently in development and cannot yet be used.
```

## Description
The idea behind this project is that for many cosmological simulation it would be convenient to have a good resolution at small scales while also have 
fairly large simulation boxes. Unfortunately, this is often computationally not feasable. <p>
Therefore, it would be good to have a tool which given a simulation at lower resolution, is able to take that simulation and generate small scale 
structures within it. <p>
This could be done in the following way:
1. Run the large simulation box requiered
2. Run two small simulation boxes, one with the resolution of the large scale simulation and another one with a better resolution.
3. Generate a training sample from the above simulations using this toolkit.
4. Train a neural network to generate the small scale structures into the simulation box with the smaller resolution using this toolkit.
5. Use the trained network to generate small scale structures into the large scale simulation.

Here is an example of the 2D projected results of a super-resolution using a neural network:
<figure>
<div class="row">
  <div class="column">
<img src="/images/lr.png" alt="drawing" width="200"/>
<img src="/images/l_all_sr.png" alt="drawing" width="200"/>
<img src="/images/hr.png" alt="drawing" width="200"/> <br>

<figcaption>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Low resolution 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp&nbsp;&nbsp;
Super resolution &nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;
High resolution</figcaption>
</figure>&nbsp;&nbsp;

## Installation
```
https://github.com/IvanKostyuk94/SuperResolution.git
cd SuperResolution
pip install .
```

## User guide
1. Copy the run directory
2. Move the low and high resolution simulation you want to use for training into the 'input_data'
3. Adjust the parameters in 'config.py'
4. Run 'run.py'
5. The trained network can now be retrieved in TBD

## Organization
- run/ -contains 'config.py' and 'run.py' should be copied in every new training directory
- src/ -contains code to build the package
  - loss/ -contains different losses. Each module should contain one set of losses for a network. E.g. for a GAN model it should contain the loss of the generator and of the critic.
  - networks/ -contains functions of the neural networks that can be trained. Each module should contain the networks for one model. E.g. one generator and one critic function for a GAN model.  
  - building_blocks/ -Contains classes to build a trainable neural networks out of the network functions, opitimizers and losses.
  - network_trainer.py -Class which builds a trainable model out of the building blocks
  - utils.py -Contains helpfull functions for preparing runs
- deprecated/ -contains deprecated code only needed for the development process
- images/ =a few example images used in this README

