using CSV
using DataFrames	
using Random
using MLDataUtils
using BenchmarkTools

import StatsBase: ProbabilityWeights
import ITEAregressor: ITEA, RMSE


df_names = ["airfoil", "cars", "concrete", "energyCooling", "energyHeating", "grossOutput",
     "qsarAquaticToxicity", "wineRed", "wineWhite", "yachtHydrodynamics"]

df_results = try
    CSV.File("../results/itea_results.csv") |> DataFrame
catch err
    DataFrame(Dataset=[], Rep=[], Time=[], Fitness_train=[], Fitness_test=[])
end

for df_name in df_names
    print("Testing the data set $df_name: ")

    df_data = CSV.File("../datasets/$(df_name).csv") |> DataFrame

    # Scramble the data, as some databases usually saves in some logical order
    df_data = df_data[Random.shuffle(1:end), :]

    # Filtering results file to see how many runs we have
    df_filtro = df_results[df_results.Dataset .== df_name, :]

    for i in 1:30
        # Let's see whether or not you have this execution data
        if size(df_filtro[df_filtro.Rep .== i, :], 1) == 0
            print("$i ")

            # Separating everything into training and testing
            train, test = splitobs(df_data, at = 0.7)
    
            # Separating training and testing into X and y
            train_X = Matrix{Float64}(train[:, 1:end-1])
            train_y = Vector{Float64}(train[:, end])
            
            test_X  = Matrix{Float64}(test[:, 1:end-1])
            test_y  = Vector{Float64}(test[:, end])

            # Saving @elapsed macro runtime and the best solution
            exec_time = @elapsed (bestsol = ITEA(
                train_X,
                train_y,
                Function[identity, sin, cos, tan, sqrt, log, exp, exp10],
                max(1, size(train_X, 2)-2),

                # named arguments (optional)
                tourn_size = 2,
                expo_bounds = (-2, 2),
                terms_bounds=(1, 5),
                popsize =100,
                gens = 100,
                mutationWeights = ProbabilityWeights([.11, .11, .11, .33, .33]),
                adjust_method ="levenberg_marquardt_adj",
            ))

            println(bestsol)

            # Saving execution results
            push!(df_results, (
                df_name,
                i,
                exec_time,
                RMSE(bestsol, train_X, train_y),
                RMSE(bestsol, test_X, test_y)
            ))
            
            # writing after each repetition
            CSV.write("../results/itea_results.csv", df_results)
            
            # Let's force the garbage collector
            GC.gc()
        end
    end

    # End for the dataset, let's go to the next
    println("Finished for dataset $(df_name).")
end