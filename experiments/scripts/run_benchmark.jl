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

"""Auxiliary function that performs a gridsearch and returns a tuple with
5 elements:
(popsize, terms_bounds, strength_bounds, gens, adj_method).

The gridsearch does save partial evaluations of different methods, and also 
reports the average performance of every configuration once it is finished
for a specific data set.
"""
function _gridsearch(df_name, adj_method)

    # set to true and create a fixed configuration if you do not intend to
    # perform a gridsearch
    if false
        return (
            250,
            (2, 15),
            (-3, 3),
            400,
            adj_method
        )
    end

    # Setting the grid to search (order is important here, do not change!)
    popsize      = [100, 250, 500]
    terms_bounds = [(2, 10), (2, 15)]
    strength_bounds  = [(-2, 2), (-3, 3)]

    # Partial configurations, still missing popsize (which is defined based on
    # popsize) and adj_method
    configs = vec(collect(
        Iterators.product(popsize, terms_bounds, strength_bounds)))

    # Adding the missing parameter values
    final_configs = []
    for i in eachindex(configs)
        popsize = configs[i][1]
        
        push!(
            final_configs, (configs[i]..., Int64(100000/popsize), adj_method))
    end

    # Loading previous results if exists. Creates a new df otherwise
    df_gridsearch = try
        CSV.File("../results/gridsearch_$(adj_method).csv") |> DataFrame
    catch err
        if isa(err, ArgumentError) 
            DataFrame(
                Dataset=String[], Fold=Int32[], Rep=Int32[],
                Exec_time=Float64[], Conf=Int32[], Error_train=Float64[],
                Error_test=Float64[], Expr=String[])
        else
            rethrow(err)
        end
    end

    # Avoid using fixed size strings for data set names 
    df_gridsearch[!, :Dataset] = convert.(String, df_gridsearch[:, :Dataset])

    println("Performing the gridsearch for $(size(final_configs, 1))...")

    for conf in eachindex(final_configs)

        # Filtering the gridsearch results to check if it is finished
        df_filtro = df_gridsearch[
            (df_gridsearch[!,:Dataset] .== df_name) .&
            (df_gridsearch[!,:Conf] .== conf), :]

        # we have 5 folds (numbered from [0, 4]) and want to have 30 executions.
        # we perform 6 executions for each fold. Loops here behaves as indexing
        # starts at 0
        for fold in 0:4
            df_train = CSV.File("../datasets/$(df_name)-train-$(fold).dat") |> DataFrame            
            df_test  = CSV.File("../datasets/$(df_name)-test-$(fold).dat")  |> DataFrame
                        
            # Separating training and testing into X and y
            train_X = Matrix{Float64}(df_train[:, 1:end-1])
            train_y = Vector{Float64}(df_train[:, end])

            test_X  = Matrix{Float64}(df_test[:, 1:end-1])
            test_y  = Vector{Float64}(df_test[:, end])

            for rep in 0:5
                # Checking if we can need to do this execution
                if size(df_filtro[
                    (df_filtro[!,:Fold] .== fold) .&
                    (df_filtro[!,:Rep] .== rep), :], 1) == 0

                    println("Executing conf $(conf) dataset $(df_name) fold $(fold) rep $(rep)")

                    exec_time, bestsol = _fit_ITEA(df_name, fold, final_configs[conf]...)

                    # Saving the results
                    push!(df_gridsearch,
                        (df_name, fold, rep, exec_time, conf, 
                        RMSE(bestsol(train_X), train_y),
                        RMSE(bestsol(test_X), test_y), to_str(bestsol)) )
                    
                    CSV.write("../results/gridsearch_$(adj_method).csv", df_gridsearch)
                    
                    GC.gc()
                end
            end
        end
    end 
    
    # Reporting in the terminal how each configuration scored
    df = df_gridsearch[df_gridsearch[!,:Dataset] .== df_name, :]
    df = @_ groupby(df, [:Dataset, :Conf]) |> combine(__, :Error => mean => :Avg_Error)

    println(df)

    return final_configs[df[argmin(df[!, :Avg_Error]), :Conf]]
end


function main()
    df_names = ["airfoil", "concrete", "energyCooling", "energyHeating",
        "wineRed", "wineWhite", "yacht", "geographical", "tecator", "towerData"]

    # executing a small experiment to force Julia to pre-compile the
    # method
    _fit_ITEA("yacht", 0, 10, (1, 3), (-1, 1), 10,
        "levenberg_marquardt_ordinary_least_squares_adj")

    for adj_method in [
        "levenberg_marquardt_ordinary_least_squares_adj",
        "levenberg_marquardt_adj", 
        "ordinary_least_squares_adj",
        "ordinary_least_squares_levenberg_marquardt_adj",
        nothing
    ]
        
        df_results = try
            CSV.File("../results/results_$(adj_method).csv") |> DataFrame
        catch err
            if isa(err, ArgumentError) 

                DataFrame(
                    Dataset=String[], Fold=Int[], Rep=Int[],
                    Exec_time=Float64[], RMSE_train=Float64[],
                    NMSE_train=Float64[], MAE_train=Float64[],
                    RMSE_test=Float64[], NMSE_test=Float64[],
                    MAE_test=Float64[], N_nodes=Int[], Expr=String[])
            else
                rethrow(err)
            end
        end

        # Avoid using fixed size strings for data set names
        df_results[!, :Dataset] = convert.(String, df_results[:, :Dataset])

        for df_name in df_names
            println("Testing the data set $df_name: ")

            best_config = _gridsearch(df_name, adj_method)

            df_filtro = df_results[df_results.Dataset .== df_name, :]

            for fold in 0:4
                df_train = CSV.File("../datasets/$(df_name)-train-$(fold).dat") |> DataFrame
                df_test  = CSV.File("../datasets/$(df_name)-test-$(fold).dat")  |> DataFrame
                
                # Separating training and testing into X and y
                train_X = Matrix{Float64}(df_train[:, 1:end-1])
                train_y = Vector{Float64}(df_train[:, end])

                test_X  = Matrix{Float64}(df_test[:, 1:end-1])
                test_y  = Vector{Float64}(df_test[:, end])

                for rep in 0:5
                    if size(df_filtro[
                        (df_filtro[!,:Fold] .== fold) .& (df_filtro[!,:Rep] .== rep), :], 1) == 0
                    
                        println("Executing dataset $(df_name) fold $(fold) rep $(rep)")

                        exec_time, bestsol = _fit_ITEA(df_name, fold, best_config...)

                        push!(df_results, (
                            df_name, fold, rep, exec_time,

                            RMSE(bestsol(train_X), train_y),
                            NMSE(bestsol(train_X), train_y),
                            MAE(bestsol(train_X), train_y),

                            RMSE(bestsol(test_X), test_y),
                            NMSE(bestsol(test_X), test_y),
                            MAE(bestsol(test_X), test_y),

                            count_nodes(bestsol), to_str(bestsol, digits=3)
                        ))
                        
                        CSV.write("../results/results_$(adj_method).csv", df_results)
                        
                        GC.gc()
                    end
                end
            end
        end
    end
end

main()