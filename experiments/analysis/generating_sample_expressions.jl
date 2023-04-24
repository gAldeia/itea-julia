using CSV
using DataFrames	
using Random
using MLDataUtils
using BenchmarkTools
using Underscores #macro @_ for passing closures to functions


import ITEAregressor: ITEA, RMSE, NMSE, MAE, to_str, count_nodes

"""Auxiliary function that takes as arguments the ITEA configuration
and performs the fit. Returns the elapsed time and the best solution found.
"""
function _fit_ITEA(
    df_name, fold, popsize, terms_bounds, strength_bounds, gens, adjust_method)
    
    df_train = CSV.File("../datasets/$(df_name)-train-$(fold).dat") |> DataFrame
    
    train_X = Matrix{Float64}(df_train[:, 1:end-1])
    train_y = Vector{Float64}(df_train[:, end])

    exec_time = @elapsed (bestsol = ITEA(
        train_X, train_y,
        ["identity", "sin", "cos", "tan", "log", "exp", "sqrt"],
        2, #max(1, size(train_X, 2)),

        popsize       = popsize,
        terms_bounds  = terms_bounds,
        strength_bounds   = strength_bounds,
        gens          = gens,
        adjust_method = adjust_method,

        mutationWeights = [.11, .11, .11, .33, .33],
        fitness_method="RMSE",
        verbose=10))

    return (exec_time, bestsol)
end

function main()
    df_names = ["airfoil", "concrete", "energyCooling", "energyHeating",
        "wineRed", "yacht"] #, "wineWhite", "geographical", "tecator", "towerData"]

    df_results = DataFrame(
                Dataset=String[], Adj_method=String[],
                Exec_time=Float64[], RMSE_train=Float64[],
                NMSE_train=Float64[], MAE_train=Float64[],
                RMSE_test=Float64[], NMSE_test=Float64[],
                MAE_test=Float64[], N_nodes=Int[], Expr=String[])

    for adj_method in [
        nothing,
        "ordinary_least_squares_adj",
        "ordinary_least_squares_levenberg_marquardt_adj",
        "levenberg_marquardt_ordinary_least_squares_adj",
        "levenberg_marquardt_adj"]
        for df_name in df_names
            println("Testing the data set $df_name: ")

            best_config = (
                250,
                (1, 4), # Small configuration, just to have presentable equations
                (-3, 3),
                400,
                adj_method
            )

            fold = 0
            df_train = CSV.File("../datasets/$(df_name)-train-$(fold).dat") |> DataFrame
            df_test  = CSV.File("../datasets/$(df_name)-test-$(fold).dat")  |> DataFrame
            
            # Separating training and testing into X and y
            train_X = Matrix{Float64}(df_train[:, 1:end-1])
            train_y = Vector{Float64}(df_train[:, end])

            test_X  = Matrix{Float64}(df_test[:, 1:end-1])
            test_y  = Vector{Float64}(df_test[:, end])

            println("Executing dataset $(df_name)")

            exec_time, bestsol = _fit_ITEA(df_name, fold, best_config...)

            push!(df_results, (
                df_name, string(adj_method), exec_time,

                RMSE(bestsol(train_X), train_y),
                NMSE(bestsol(train_X), train_y),
                MAE(bestsol(train_X), train_y),

                RMSE(bestsol(test_X), test_y),
                NMSE(bestsol(test_X), test_y),
                MAE(bestsol(test_X), test_y),

                count_nodes(bestsol), to_str(bestsol, digits=3)
            ))
            
            CSV.write("./sample_expressions.csv", df_results)
        
            GC.gc()
        end
    end

    println(df_results)

                        
end

main()