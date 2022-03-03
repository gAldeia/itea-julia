__precompile__()

module ITEAregressor

const ITEA_CACHE_SIZE = 3000

# TODO: usar mais símbolos e nomes significativos e curtos
# TODO: colocar o tipo de retorno em cada função
# TODO: descartar coeficientes 0 (intercepto e termos)
# TODO:python wrapper
# TODO: exportar tudo que deve ser visível
# TODO: desenvolver e testar melhor os códigos
# TODO: respeitar régua na coluna 80 nos códigos
# TODO: resolver bug import StatsBase, unificar random para StatsBase
# TODO: AVISAR QUE o ITEA não pode ser paralelizado para diferentes bases em um mesmo processo (paralelismo do ITEA só em diferentes processos, pois há uma memória compartilhada entre threads)

export
    ITEA, # Interaction-Transformation Evolutionary Algorithm
    evaluate,  # evaluate and predict methods
    R2neg, MAE, MSE, NMSE, RMSE, # regression metrics
    to_str, count_nodes # utility methods

import Base

using LinearAlgebra: pinv, LAPACKException, SingularException
using ForwardDiff: derivative
using LsqFit: curve_fit
using StatsBase: sample, ProbabilityWeights, randperm, rle
using Distributions: Uniform

using Printf # verbose 

# thread-safe memoization (https://github.com/JuliaCollections/LRUCache.jl)
using LRUCache 

# Multithread (shared memory)
using Folds

using Statistics: mean, median, std, var

include("ITexpr.jl")
include("Utils.jl")
include("Evaluation.jl")
include("ParameterOptimization.jl")
include("Fitness.jl")
include("EvolutionaryAlgorithm.jl")

end # module