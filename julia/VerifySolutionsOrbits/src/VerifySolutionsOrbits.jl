# Startup script. To be updated later to a module.

# module VerifySolutionsOrbits
using JLD2, LinearAlgebra, ForwardDiff, RadiiPolynomial, CSV, DataFrames, DelimitedFiles
include("proofs.jl")
include("controllers.jl")
include("CartPoleSwingup.jl")
include("Pendulum.jl")
include("numerics.jl") 
include("batch.jl")
include("network.jl")
include("integrator.jl")

# export prove_transients_pendulum, prove_transients_cartpole, prove_transients_cartpole_NeuralNet, proof_cartpole_LargeNet       # Batch proofs
# export Landajuela_proofs_LaTeXString, output_LaTeX_table_pendulum, output_LaTeX_table_cartpole      # Tables (and batch Landajuela proofs)
# export load_tensors, load_tensors_zero_bias, convert_weight_bias_bigfloat, Network      # Neural network

# end