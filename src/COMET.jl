module COMET

using Flux
using Functors
using Zygote
using OrdinaryDiffEq
using SciMLBase
using Parameters
using ProgressMeter
using Printf
using LinearAlgebra
using CUDA

# Include the other files
include("models.jl")
include("training.jl")
include("utils.jl")

export COMETModel, MetaCOMETModel, train!,UnifiedModel, select_model

end
