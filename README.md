### code for ECAI23 paper entitled
### _Worrisome Properties of Neural Network Controllers and Their Symbolic Representations_ by Jacek Cyranka, Kevin E M Church and Jean-Philippe Lessard

Codebase consists of two separate packages (Python and Julia)

Raw csv datafiles with controller data imported in the paper appendix are found in 
python/pendulum and python/cartpole_swingup directories corresponding to the
respective controller class.

_Controllers Data_

Directory per studied problem. Each controller class in its  Csv datafiles containing info on the studied controllers, and the computed metrics reported in the paper (like average return & penalty).   

1) ReLU... contain data of the deepNN controller,
2) SmallNet contain data of the ditilled SmallNets
3) The 'finetune' folders contain data with the finetuned controllers.
4) unstabilized... folders contain the hall-of-fames of the found unstabilized solutions and potential orbits.

_Python micro-documentation_

Requirements are :
*pytorch
*pysr
*pandas
*matplotlib
*gym
*sympy
*numpy
*scikit-learn
*sb3
*sb3-zoo (for training)

The workflow goes as follows:

1) Train a deep NN controller using the stable-baselines3  https://github.com/DLR-RM/rl-baselines3-zoo;, and put the checkpoint in the working dir; We include the pretrained agents in the respective folders;
2) run derive_symbolic.py script (the first argument is the dir with the trained sb3 checkpoint); it will output a hall-of-fame csv file with found symbolic controllers; 
3) Then the symbolic controllers need to be defined as PyTorch parametrized functions and serialized in .pt file, see the example in pendulum_symbolic2.py and cartpole_swingup_symbolic2.pywingup_; We include the serialized pytorch symbolic controllers in the respective folders;
4) To distill a small NN use smallNN_distill.py script (the first argument is the dir with the trained sb3 checkpoint);
5) For testing of the controllers contained in hall-of-fame csv (i.e. computing the average rewards for different discretizations) use pysr_controller_test_hof.py script ;
6) Finally apply the fine-tuning script (pendulum/swingup)_cma.py;
7) The unstabilized solutions search is implemented in pendulum_transientsCMA.py and swingup_transientsCMA.py for pendulum and swingup systems respectively;

_Julia micro-documentation._

- Use Pkg to activate/instantiate the environment, then include all required components as follows...
julia> import Pkg
julia> Pkg.activate("path/to/julia/VerifySolutionsOrbits") # edit the path accordingly
julia> Pkg.instantiate()
julia> include("path/to/julia/VerifySolutionsOrbits/src/VerifySolutionsOrbits.jl");

*NOTE/HELP TO WINDOWS USERS, if julia is giving you errors when trying to copy and paste paths, see this post: 
https://discourse.julialang.org/t/windows-file-path-string-slash-direction-best-way-to-copy-paste/29204

- Note: the CAP functionality has been implemented to be (mostly) readable rather than fast, with a focus on ensuring that the low-level functions
for running integrators and generating the maps G and DG (written F and DF in this code) are fairly general. The downside is that there are some type 
instabilities that are unavoidable without significantly more time investment, and this can make the proofs run slower than they should. 
This is especially the case for anything that communicates with a data file. The symbolic controllers are especially problematic, since each row of the csv
file requires the controller (and subsequent components) to be recompiled. It would be far more efficient (and transparent, in terms of the code) to use 
pyCall to grab fields out of the pytorch structures directly and then to parameterize the controllers on the julia side. We are looking into doing that.

Usage Example 1: Loading the Large NN, finding and proving all periodic orbits in cartpole swingup model. This has been scripted for ease of reproduction.
-- Run the following from the REPL: 
julia> proof_cartpole_LargeNet(file,folder);
where file = suitably formatted (string) path to: ...\julia\VerifySolutionsOrbits\orbits_400_300_ReLU_cartpole\relu_400_300_cartpole_swingup.jld2
where folder = suitably formatted (string) path to: ...\controllers_data\cartpole_swingup\ReLU_400_300

Usage Example 2: Loading the Small NN, proving all unstabilized solutions and reconstructing the table (LaTeX code) from the paper.
-- Import weights and biases. 
julia> W,B = load_tensors_zero_bias(path,2);
where path = suitably formatted (string) path to:   ...\controllers_data\pendulum\SmallNet
-- Convert weights and biases to BigFloat (4096 bits; required because of extreme wrapping effect). 
julia> W,B = convert_weight_bias_bigfloat(W,B;precision=4096);
-- Pass the network to the batch proof function.
julia> activations = ["ReLU","Tanh"];
julia> scaling(x) = x;
julia> solutions, penalty, escape, escape_flag, step_size = prove_transients_cartpole_NeuralNet(x->Network(x,W,B,activations,scaling),csv_path);
where csv_path = suitably formatted (string) path to: ...\controllers_data\cartpole_swingup\transients_infiniteray_cma\SmallNet25_cma_swingup_final_model_transients_hof.csv
-- Print the LaTeX table
julia> str = output_LaTeX_table_cartpole("Small NN", csv_path, solutions, penalty, escape_flag);
where csv_path = suitably formatted (string) path to: ...\controllers_data\cartpole_swingup\transients_infiniteray_cma\SmallNet25_cma_swingup_final_model_transients_hof.csv
julia> println(str)
