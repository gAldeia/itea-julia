__precompile__()


module ITEAregressor


# TODO: exportar tudo que deve ser visível

# TODO: desenvolver e testar melhor os códigos

# TODO: resolver bug import StatsBase, unificar random para StatsBase

import Base

using LinearAlgebra: pinv
using ForwardDiff: derivative
using LsqFit: curve_fit
using StatsBase: sample, ProbabilityWeights, randperm

using Statistics

# TODO:python wrapper

include("ITexpr.jl")
include("Utils.jl")
include("Evaluation.jl")
include("ParameterOptimization.jl")
include("Metrics.jl")
include("EvolutionaryAlgorithm.jl")


greet() = print("Hello World!")


end # module
