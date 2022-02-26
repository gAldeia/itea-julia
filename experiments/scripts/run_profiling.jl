using CSV
using DataFrames	
using Random
using MLDataUtils
using BenchmarkTools

using Profile, PProf

import StatsBase: ProbabilityWeights
import ITEAregressor: ITEA, RMSE


df_name = "airfoil"

df_data = CSV.File("../datasets/$(df_name).csv") |> DataFrame

# Scramble the data, as some databases usually saves in some logical order
df_data = df_data[Random.shuffle(1:end), :]

# Separating everything into training and testing
train, test = splitobs(df_data, at = 0.7)

# Separating training and testing into X and y
train_X = Matrix{Float64}(train[:, 1:end-1])
train_y = Vector{Float64}(train[:, end])

test_X  = Matrix{Float64}(test[:, 1:end-1])
test_y  = Vector{Float64}(test[:, end])

# Função de experimento
experiment = (; popsize=100, gens=100) -> ITEA(
    train_X,
    train_y,
    Function[identity, sin, cos, tan, sqrt, log, exp, exp10],
    max(1, size(train_X, 2)-2),

    # named arguments (optional)
    tourn_size = 2,
    expo_bounds = (-2, 2),
    terms_bounds=(1, 5),
    popsize =popsize,
    gens = gens,
    mutationWeights = ProbabilityWeights([.11, .11, .11, .33, .33]),
    adjust_method ="levenberg_marquardt_adj",
)

# Ignoramos a primeira pois julia compila no primeiro uso. Vamos fazer para dar 
# um trigger na pré compilação
experiment(popsize=20, gens=20)

@profile experiment()
pprof(;webport=5000)

# Segurando o profile
if !isinteractive()
    wait(Condition())
end