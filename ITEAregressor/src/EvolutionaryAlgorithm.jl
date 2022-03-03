function _init_population(
    popsize::Int64, nvars::Int64, transf_funcs:: Vector{Function},
    expo_bounds::Tuple{Int64, Int64}, terms_bounds::Tuple{Int64, Int64},
    max_nonzero_k::Int64)

    (min_expo,  max_expo ) = expo_bounds
    (min_terms, max_terms) = terms_bounds

    ITpop(map(collect(1:popsize)) do _
        n_terms = rand(min_terms:max_terms)
        
        gs = rand(transf_funcs, n_terms)
        ks = zeros(n_terms, nvars) # we need to create ks as floats

        for i in 1:n_terms
            # Selecting exponents that will be != 0, with at least 1 
            # exponent selected
            nonzero_k = rand(1:nvars, rand(1:max_nonzero_k))

            # Skipping zero. if min_expo is not negative this still work.
            # Validation of the parameters must check that the range(min, max)
            # contains at least one element different from zero and min <= max.
            # this guarantees that it will never be created an empty expression
            ks[i, nonzero_k] = rand(
                (min_expo:min(-1, max_expo)..., max(1, min_expo):max_expo...), 
                size(nonzero_k, 1)
            )
        end

        ITexpr([IT(gs[i], ks[i, :]) for i in 1:n_terms])
    end)
end


function _mutate_population(
    pop::ITpop, nvars::Int64, probabilities::ProbabilityWeights,
    transf_funcs:: Vector{Function},
    expo_bounds::Tuple{Int64, Int64}, terms_bounds::Tuple{Int64, Int64},
    max_nonzero_k::Int64)

    # probabilities tem que ter 5 elementos que somados dão 1, representando
    # a probabilidade de cada mutação
    
    (min_expo,  max_expo ) = expo_bounds
    (min_terms, max_terms) = terms_bounds

    ITpop(map(pop.ITexprs) do itexpr

        # Changing probabilities of invalid mutations
        # (they will invalidate the expr size bounds)
        adjusted_probabilities = copy(probabilities)

        if size(itexpr.ITs, 1) >= max_terms
            adjusted_probabilities[1:3] .= 0.000
        end

        if size(itexpr.ITs, 1) <= min_terms
            adjusted_probabilities[4] = 0.000
        end

        # Ao menos uma mutação vai sempre existir, que é a dos expoentes!

        # Selecting the mutation that will be applied
        mutation_id = sample(1:5, ProbabilityWeights(adjusted_probabilities))
        
        if mutation_id==1 # add
            g = rand(transf_funcs)
            k = zeros(nvars)
    
            nonzero_k = rand(1:nvars, rand(1:max_nonzero_k))
            k[nonzero_k] = rand(
                (min_expo:min(-1, max_expo)..., max(1, min_expo):max_expo...), 
                size(nonzero_k, 1)
            )

            ITexpr([[IT(it.g, copy(it.k)) for it in itexpr.ITs]..., IT(g, k)])

        elseif mutation_id==2 || mutation_id==3 # 2 - interp, 3 - intern
            n_terms = size(itexpr.ITs, 1)

            idx1, idx2 = rand(1:n_terms, 2)

            k1, k2 = itexpr.ITs[idx1].k, itexpr.ITs[idx2].k
    
            # Vamos ver quantos não nulos tem...
            nonzero_k = [i for i in 1:nvars if k1[i] != 0 || k2[i]!=0]
            
            # ...e pegar no máximo max_nonzero_k aleatórios para usar
            nonzero_k = nonzero_k[
                randperm(size(nonzero_k, 1))[1:min(size(nonzero_k, 1), max_nonzero_k)] ]
            
            # Vamos pegar a função de um deles, que seja do primeiro
            g = itexpr.ITs[idx1].g

            k = zeros(nvars)
            
            # Usando min e max para controlar os expoentes dentro do intervalo permitido
            if mutation_id==2
                k[nonzero_k] = min.(max.(k1[nonzero_k] .+ k2[nonzero_k], min_expo), max_expo)
            else # mutation_id==3
                k[nonzero_k] = min.(max.(k1[nonzero_k] .- k2[nonzero_k], min_expo), max_expo)
            end

            ITexpr([[IT(it.g, copy(it.k)) for it in itexpr.ITs]..., IT(g, k)])

        elseif mutation_id==4 # drop
            n_terms = size(itexpr.ITs, 1)
            idx = rand(1:n_terms)

            ITs = [IT(itexpr.ITs[i].g, copy(itexpr.ITs[i].k)) for i in 1:n_terms if i != idx]
    
            ITexpr(ITs)
        else # interaction strengths
            n_terms = size(itexpr.ITs, 1)
            idx = rand(1:n_terms)

            g, k = itexpr.ITs[idx].g, copy(itexpr.ITs[idx].k)

            # Vamos pegar um índice de expoente e sortear um novo valor que
            # não seja o mesmo que já estava antes e respeite os limites
            idx_k = rand(1:nvars)
            k[idx_k] = rand( [i for i in min_expo:max_expo if i != k[idx_k] ] )

            # Vamos ver quantos não nulos tem...
            nonzero_k = [i for i in 1:nvars if k[i] != 0]
        
            # Se ainda não está no limite máximo podemos mutacionar sem se
            # preocupar. Se não, vamos zerar alguém pra compensar o aumento no 
            # número de não nulos
            if size(nonzero_k, 1) > max_nonzero_k
                # ...e pegar no máximo max_nonzero_k aleatórios para usar
                nonzero_k = nonzero_k[
                    randperm(size(nonzero_k, 1))[1:min(size(nonzero_k, 1), max_nonzero_k)] ]
            
                # Vamos criar um vetor de zeros e copiar apenas o máximo
                # de não nulos permitidos
                new_k = zeros(nvars)
                new_k[nonzero_k] = k[nonzero_k]

                k = new_k
            end

            ITs = [IT(itexpr.ITs[i].g, copy(itexpr.ITs[i].k)) for i in 1:n_terms if i != idx]

            ITexpr([ITs..., IT(g, k)])
        end
    end)
end

function ITEA(
    # Obrigatory arguments
    X::Array{T, 2},
    y::Array{T, 1},
    transf_funcs::Array{String, 1},
    max_nonzero_k::Int;

    # Optional configurable arguments 
    popsize::Int = 250,
    gens::Int    = 250,

    expo_bounds::Tuple{Int, Int}  = (-2, 2),
    terms_bounds::Tuple{Int, Int} = (1, 5),

    mutationWeights::Array{T, 1} = [.11, .11, .11, .33, .33],

    adjust_method::Union{String, Nothing} = nothing, 
    fit_intercept::Bool = true,

    fitness_method::String = "RMSE",
    tourn_size::Int = 3,

    # Execution behavior
    verbose::Int = 1,
    threaded_execution::Bool=true) where {T<:Number}
    
    # Documentar: o que é paralelizado e memorizado: fitness e coeficientes.
    # deve iniciar julia com mais threads para usar isso de maneira efetiva

    @assert terms_bounds[1] >= 1 "The minimum number of terms should be greater than zero"
    @assert terms_bounds[2] >= terms_bounds[1] "The maximum number of terms should be at least equal to the mininum"
    @assert size(mutationWeights, 1) == 5 "There are 5 mutations. You need to give 5 mutation weights"
    @assert expo_bounds[1] <= expo_bounds[2] "The lower bound must be smaller or equal to the upper bound"
    @assert 0 in collect(expo_bounds[1]:expo_bounds[2]) "The number 0 must be within the exponents range"

    # Converting to probability weights
    mutationWeights = ProbabilityWeights(mutationWeights)

    # Converting function names as strings to actual functions
    transf_funcs = [eval(Symbol(f)) for f in transf_funcs]

    nvars = size(X, 2)

    # initializing population
    pop = _init_population(
        popsize, nvars, transf_funcs, expo_bounds, terms_bounds, max_nonzero_k)
    
    # Adjusting
    pop = if threaded_execution
        ITpop(Folds.map(
            p -> free_params_adj(
            p, X, y; method=adjust_method, fit_intercept=fit_intercept),
            pop.ITexprs))
    else
        ITpop([free_params_adj(
            p, X, y; method=adjust_method, fit_intercept=fit_intercept)
            for p in pop.ITexprs])
    end

    fitnesses = if threaded_execution
        [fitness(itexpr, X, y, metric=fitness_method) for itexpr in pop.ITexprs]
    else
        Folds.map(
            itexpr -> fitness(itexpr, X, y, metric=fitness_method),
            pop.ITexprs)
    end

    if verbose>0
        if threaded_execution
            if Threads.nthreads()==1
                @warn "ITEA is set to run in multiple threads, but Julia " *
                       "was executed with only 1 thread. To use multiple " * 
                       "threads, run julia passing --threads <n>. To disable " *
                       "this warning, if julia is not executed with multiple " * 
                       "threads, then call ITEA with threaded_execution=false."
            else
                @info "Executing ITEA with $(Threads.nthreads()) threads."
            end
        end

        @printf("\n+------+-----------+-----------+------------+----------+\n")
        @printf(  "|  Gen |  Best Fit |  Avg Fit  | Smlst Size | Avg Size |\n")
        @printf(  "+------+-----------+-----------+------------+----------+\n")
    end

    for g in 1:gens
        if verbose>0 && mod(g, verbose) == 0
            sizes = [count_nodes(p) for p in pop.ITexprs]
            avg_fitness = min(mean(fitnesses), 9.999e+99)
            @printf("| %4d | %.3e | %.3e | %10d | %8.3f |\n",
                g, minimum(fitnesses), avg_fitness, minimum(sizes), mean(sizes))
        end

        xmen = _mutate_population(
            pop, nvars, mutationWeights, transf_funcs,
            expo_bounds, terms_bounds, max_nonzero_k)

        xmen = if threaded_execution
            ITpop(Folds.map(
                xm -> free_params_adj(
                    xm, X, y; method=adjust_method, fit_intercept=fit_intercept),
                xmen.ITexprs))
        else
            ITpop([free_params_adj(
                xm, X, y; method=adjust_method, fit_intercept=fit_intercept)
                for xm in xmen.ITexprs])
        end 

        # Including mutated children. Population now has doubled in size.
        pop = ITpop([pop.ITexprs..., xmen.ITexprs...])

        fitnesses = if threaded_execution
            [fitnesses..., Folds.map(
                xm -> fitness(xm, X, y, metric=fitness_method),
                xmen.ITexprs)...]
        else
            [fitnesses...,
                [fitness(itexpr, X, y, metric=fitness_method) for itexpr in xmen.ITexprs]...]
        end
    
        # performing tournament selections until a new population is created
        winner_idxs = map(1:popsize) do i
            # Select random individuals to a tournament of size (tourn_size)
            idxs = rand(1:popsize*2, tourn_size)

            # Returns the index in the original population for the one with best fitness 
            idxs[argmin(fitnesses[idxs])]
        end

        # Saving the fitness to avoid re-evaluation
        individuals, fitnesses = pop.ITexprs[winner_idxs], fitnesses[winner_idxs]

        pop = ITpop(individuals)
    end

    clear_fitness_memoize()
    clear_adj_memoize()

    if verbose>0
        @printf("+------+-----------+-----------+------------+----------+\n")
    end

    pop.ITexprs[argmin(fitnesses)]
end