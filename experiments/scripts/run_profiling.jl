using CSV
using DataFrames	
using Random
using MLDataUtils
using BenchmarkTools

using Profile, PProf

import StatsBase: ProbabilityWeights
import ITEAregressor: ITEA, RMSE


# We'll use airfoil for profiling
df_data = CSV.File("../datasets/airfoil-train-0.dat") |> DataFrame

# Suffle the data, as some databases usually saves in some logical order
df_data = df_data[Random.shuffle(1:end), :]

# Separating everything into training and testing by creating a list of indexes
train, test = splitobs(df_data, at = 0.7)

# Separating training and testing into X and y
train_X = Matrix{Float64}(train[:, 1:end-1])
train_y = Vector{Float64}(train[:, end])

test_X  = Matrix{Float64}(test[:, 1:end-1])
test_y  = Vector{Float64}(test[:, end])

# Function to perform the experiment
experiment = (; popsize=100, gens=100) -> bestsol = ITEA(
    train_X, train_y,
    ["identity", "sin", "cos", "tan", "log", "exp", "sqrt"], #"sqrtabs",
    
    # Maximum number of non-zero strengths
    2, #max(1, size(train_X, 2)),

    popsize       = popsize,
    terms_bounds  = terms_bounds,
    expo_bounds   = expo_bounds,
    gens          = gens,
    adjust_method = adjust_method,

    mutationWeights = [.11, .11, .11, .33, .33],
    fitness_method="RMSE",
    verbose=10)

# Ignoring the first execution, since julia compiles the method when 
# it's first called. This execution triggers the compilation
experiment(popsize=20, gens=20)

# Now we do the profiling
@profile experiment()
pprof(;webport=5000)

if !isinteractive()
    wait(Condition())
end