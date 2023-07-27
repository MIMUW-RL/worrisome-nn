### code release for ECAI23 paper entitled
### _Worrisome Properties of Neural Network Controllers and Their Symbolic Representations_ by Jacek Cyranka, Kevin E M Church and Jean-Philippe Lessard

Codebase consists of two separate packages (Python and Julia).

Controllers training, sybolic/small-net regression and persistent solutions search is done in Python,
whereas computer-assisted proofs are performed in Julia due to availability of convenient libraries, 
weights of controllers including networks are transferred through csv files.

Raw csv datafiles with controller data imported in the paper appendix are found in 
[controllers_data/pendulum](controllers_data/pendulum) and [controllers_data/cartpole_swingup](controllers_data/cartpole_swingup) directories corresponding to the respective controller class.

### Studied problems

### Controllers Data

Each directory corresponding to the studied controller class contains info on the studied controllers, the computed metrics reported in the paper (like average return & penalty).   

1) ReLU... contain data of the deepNN controller (e.g., [controllers_data/pendulum/ReLU_256_256_256_simple2](controllers_data/pendulum/ReLU_256_256_256_simple2)),
2) SmallNet contain data of the ditilled SmallNets (e.g., [controllers_data/pendulum/SmallNet](controllers_data/pendulum/SmallNet))
3) The 'finetune' folders contain data with the finetuned controllers. (e.g., [controllers_data/pendulum/Symbolic_finetune_CMA](controllers_data/pendulum/Symbolic_finetune_CMA) ...)
4) unstabilized... folders contain the found unstabilized solutions and potential orbits listed as a hall-of-fames [controllers_data/pendulum/Symbolic_unstabilized_CMA](controllers_data/pendulum/Symbolic_unstabilized_CMA).

### Python micro-documentation

Requirements are :
* pytorch
* pysr
* pandas
* matplotlib
* gym
* sympy
* numpy
* scikit-learn
* sb3 (for RL training)
* sb3-zoo (for RL training)

The workflow:

1) Train a deep NN controller using the stable-baselines3  https://github.com/DLR-RM/rl-baselines3-zoo;, and put the checkpoint in the working dir; We include the pretrained agents in the respective folders;
2) run [python/derive_symbolic.py](python/derive_symbolic.py) script (the first argument is the dir with the trained sb3 checkpoint); it will output a hall-of-fame csv file with found symbolic controllers; 
3) Then the found symbolic controllers need to be defined as PyTorch parametrized functions and serialized in .pt file, see the example in [python/pendulum_symbolic2.py](python/pendulum_symbolic2.py); We include the serialized (pickle) pytorch symbolic controllers in the respective folders;
4) To distill a small NN use [python/smallNN_distill.py](python/smallNN_distill.py) script (the first argument is the dir with the trained sb3 checkpoint);
5) For testing of the controllers contained in a hall-of-fame csv (like computing the average rewards for different discretizations) use [python/pysr_controller_test_hof.py](python/pysr_controller_test_hof.py) script ;
6) Finally apply the fine-tuning script [python/pendulum_cma.py](python/pendulum_cma.py) and [python/swingup_cma.py](python/swingup_cma.py) for the pendulum and cartpole-swingup problem respectively;
7) The unstabilized solutions search is implemented in [python/pendulum_transientsCMA.py](python/pendulum_transientsCMA.py) and [python/swingup_transientsCMA.py](python/swingup_transientsCMA.py) for the pendulum and cartpole-swingup problem respectively;

### Julia micro-documentation

- Use Pkg to activate/instantiate the environment, then include all required components as follows...
```
julia> import Pkg
julia> Pkg.activate("path/to/julia/VerifySolutionsOrbits") # edit the path accordingly
julia> Pkg.instantiate()
julia> include("path/to/julia/VerifySolutionsOrbits/src/VerifySolutionsOrbits.jl");
```

- NOTE/HELP TO WINDOWS USERS, if julia is giving you errors when trying to copy and paste paths, see this post: 
https://discourse.julialang.org/t/windows-file-path-string-slash-direction-best-way-to-copy-paste/29204

- Note: the CAP functionality has been implemented to be (mostly) readable rather than fast, with a focus on ensuring that the low-level functions
for running integrators and generating the maps G and DG (written F and DF in this code) are fairly general. The downside is that there are some type 
instabilities that are unavoidable without significantly more time investment, and this can make the proofs run slower than they should. 
This is especially the case for anything that communicates with a data file. The symbolic controllers are especially problematic, since each row of the csv
file requires the controller (and subsequent components) to be recompiled. It would be far more efficient (and transparent, in terms of the code) to use 
pyCall to grab fields out of the pytorch structures directly and then to parameterize the controllers on the julia side. We are looking into doing that.

_Usage Example 1:_ Loading the Large NN, finding and proving all periodic orbits in cartpole swingup model. This has been scripted for ease of reproduction.

- Run the following from the REPL: 
`julia> proof_cartpole_LargeNet(file,folder);`
where file = suitably formatted (string) path to: `...\julia\VerifySolutionsOrbits\orbits_400_300_ReLU_cartpole\relu_400_300_cartpole_swingup.jld2`
where folder = suitably formatted (string) path to: `...\controllers_data\cartpole_swingup\ReLU_400_300`

_Usage Example 2:_ Loading the Small NN, proving all unstabilized solutions and reconstructing the table (LaTeX code) from the paper.

- Import weights and biases. 
`julia> W,B = load_tensors_zero_bias(path,2);`
where path = suitably formatted (string) path to:   `...\controllers_data\pendulum\SmallNet`

- Convert weights and biases to BigFloat (4096 bits; required because of extreme wrapping effect). 
`julia> W,B = convert_weight_bias_bigfloat(W,B;precision=4096);`

- Pass the network to the batch proof function.
```
julia> activations = ["ReLU","Tanh"];
julia> scaling(x) = x;
julia> solutions, penalty, escape, escape_flag, step_size = prove_transients_cartpole_NeuralNet(x->Network(x,W,B,activations,scaling),csv_path);
```
where csv_path = suitably formatted (string) path to: `...\controllers_data\cartpole_swingup\transients_infiniteray_cma\SmallNet25_cma_swingup_final_model_transients_hof.csv`

- Print the LaTeX table
`julia> str = output_LaTeX_table_cartpole("Small NN", csv_path, solutions, penalty, escape_flag);`
where csv_path = suitably formatted (string) path to: `...\controllers_data\cartpole_swingup\transients_infiniteray_cma\SmallNet25_cma_swingup_final_model_transients_hof.csv
julia> println(str)`
