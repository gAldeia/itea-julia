using CSV
using DataFrames	
using Random
using MLDataUtils
using BenchmarkTools
using Underscores #macro @_ for passing closures to functions


import ITEAregressor: ITEA, RMSE, NMSE, MAE, to_str, count_nodes


function _fit_ITEA(df_name, fold, popsize, terms_bounds, expo_bounds, gens, adjust_method)
    df_train = CSV.File("../datasets/$(df_name)-train-$(fold).dat") |> DataFrame
    
    train_X, train_y = Matrix{Float64}(df_train[:, 1:end-1]), Vector{Float64}(df_train[:, end])

    exec_time = @elapsed (bestsol = ITEA(
        train_X, train_y,
        ["identity","sin","cos","tan","sqrtabs","log","exp","abs"],
        max(1, size(train_X, 2)),

        popsize       = popsize,
        terms_bounds  = terms_bounds,
        expo_bounds   = expo_bounds,
        gens          = gens,
        adjust_method = adjust_method,

        mutationWeights = [.11, .11, .11, .33, .33],
        fitness_method="RMSE",
        verbose=10))

    return (exec_time, bestsol)
end


function _gridsearch(df_name, adj_method)

    # Criando as configurações (eu salvo o índice delas)
    popsize      = [100, 250, 500]
    terms_bounds = [(2, 10), (2, 15)]
    expo_bounds  = [(-3, 3), (-2, 2)]

    configs = vec(collect(Iterators.product(popsize, terms_bounds, expo_bounds)))

    final_configs = []
    for i in eachindex(configs)
        popsize = configs[i][1] # popsize é o primeiro element de configs
        
        push!( # Adicionando gens e o método de ajuste
            final_configs, (configs[i]..., Int64(100000/popsize), adj_method))
    end

    # Loading previous results
    df_gridsearch = try
        CSV.File("../results/gridsearch_$(adj_method).csv") |> DataFrame
    catch err
        if isa(err, ArgumentError) 
            DataFrame(Dataset=String[], Fold=Int32[], Rep=Int32[], Conf=Int32[], Error=Float64[])
        else
            rethrow(err)
        end
    end

    # Avoid using fixed size strings for data set names
    df_gridsearch[!, :Dataset] = convert.(String, df_gridsearch[:, :Dataset])

    println("Performing the gridsearch for $(size(final_configs, 1)) configs...")

    for conf in eachindex(final_configs)

        df_filtro = df_gridsearch[(df_gridsearch[!,:Dataset] .== df_name) .& (df_gridsearch[!,:Conf] .== conf), :]

        # Um dataset é dividido em 5 folds. Repetiremos 6x cada fold para obter
        # 30 execuções de uma validação cruzada
        for fold in 0:4
            df_test = CSV.File("../datasets/$(df_name)-test-$(fold).dat")  |> DataFrame
            
            test_X, test_y = Matrix{Float64}(df_test[:, 1:end-1]), Vector{Float64}(df_test[:, end])

            for rep in 0:5
                if size(df_filtro[(df_filtro[!,:Fold] .== fold) .& (df_filtro[!,:Rep] .== rep), :], 1) == 0

                    println("Executing conf $(conf) dataset $(df_name) fold $(fold) rep $(rep)")

                    exec_time, bestsol = _fit_ITEA(df_name, fold, final_configs[conf]...)

                    push!(df_gridsearch, (df_name, fold, rep, conf, RMSE(bestsol(test_X), test_y)) )
                    
                    CSV.write("../results/gridsearch_$(adj_method).csv", df_gridsearch)
                    
                    GC.gc()
                end
            end
        end
    end 
    
    df = df_gridsearch[df_gridsearch[!,:Dataset] .== df_name, :]
    df = @_ groupby(df, [:Dataset, :Conf]) |> combine(__, :Error => mean => :Avg_Error)

    println(df)

    return final_configs[df[argmin(df[!, :Avg_Error]), :Conf]]
end


function main()
    df_names = ["airfoil", "concrete", "energyCooling", "energyHeating",
        "geographical", "tecator", "towerData", "wineRed", "wineWhite", "yacht"]

    # rodar primeiro 1x sem salvar (julia pré compila nessas horas, pode enviesar medir tempo)
    _fit_ITEA("yacht", 0, 10, (1, 3), (-1, 1), 10, nothing)

    for adj_method in ["gradient_descent_adj", "levenberg_marquardt_adj", "ordinary_least_squares_adj", nothing]
        
        df_results = try
            CSV.File("../results/results_$(adj_method).csv") |> DataFrame
        catch err
            if isa(err, ArgumentError) 

                DataFrame(Dataset=String[], Fold=Int[], Rep=Int[], Time=Float64[],
                        RMSE_test=Float64[], NMSE_test=Float64[], MAE_test=Float64[],
                        Expr=String[], N_nodes = Int[])
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
                
                df_test  = CSV.File("../datasets/$(df_name)-test-$(fold).dat")  |> DataFrame
                
                test_X  = Matrix{Float64}(df_test[:, 1:end-1])
                test_y  = Vector{Float64}(df_test[:, end])

                for rep in 0:5
                    if size(df_filtro[
                        (df_filtro[!,:Fold] .== fold) .& (df_filtro[!,:Rep] .== rep), :], 1) == 0
                    
                        println("Executing dataset $(df_name) fold $(fold) rep $(rep)")

                        exec_time, bestsol = _fit_ITEA(df_name, fold, best_config...)

                        # TODO: resolver bug string is too large

                        push!(df_results, (
                            df_name, fold, rep, exec_time,

                            RMSE(bestsol(test_X), test_y),
                            NMSE(bestsol(test_X), test_y),
                            MAE(bestsol(test_X), test_y),

                            to_str(bestsol, digits=3), count_nodes(bestsol)
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