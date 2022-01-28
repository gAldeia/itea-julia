function _init_population(
    popsize::Int64, nvars::Int64, transf_funcs:: Vector{Function},
    expo_bounds::Tuple{Int64, Int64}, terms_bounds::Tuple{Int64, Int64},
    max_nonzero_k::Int64)

    # TODO: validar os intervalos de terms_bounds
    (min_expo,  max_expo ) = expo_bounds
    (min_terms, max_terms) = terms_bounds

    ITpop(map(collect(1:popsize)) do _
        n_terms = rand(min_terms:max_terms)
        
        gs = rand(transf_funcs, n_terms)
        ks = zeros(n_terms, nvars)

        for i in 1:n_terms
            # Selecting exponents that will be != 0
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

# TODO: mutação copiar coeficientes

# TODO: X e y no ITEA podem ser inteiros (deve ser subtype de number)
function _mutate_population(
    pop::ITpop, nvars::Int64, probabilities::ProbabilityWeights,
    transf_funcs:: Vector{Function},
    expo_bounds::Tuple{Int64, Int64}, terms_bounds::Tuple{Int64, Int64},
    max_nonzero_k::Int64)

    # probabilities tem que ter 5 elementos que somados dão 1, representando
    # a probabilidade de cada mutação

    # TODO: melhorar código de mutação
    
    (min_expo,  max_expo ) = expo_bounds
    (min_terms, max_terms) = terms_bounds

    ITpop(map(pop.ITexprs) do itexpr
        
        adjusted_probabilities = copy(probabilities)

        # expression will not increase any further
        if size(itexpr.ITs, 1) == max_terms
            adjusted_probabilities[1:3] .= 0.0
        end

        if size(itexpr.ITs, 1) == min_terms
            adjusted_probabilities[5] = 0.0
        end

        # Ao menos uma mutação vai sempre existir, que é a dos expoentes

        mutation_id = sample(1:5, ProbabilityWeights(adjusted_probabilities))
        
        if mutation_id==1 # add
            g = rand(transf_funcs)
            k = zeros(nvars)
    
            nonzero_k = rand(1:nvars, rand(1:max_nonzero_k))
            k[nonzero_k] = rand(
                (min_expo:min(-1, max_expo)..., max(1, min_expo):max_expo...), 
                size(nonzero_k, 1)
            )

            ITs = map(itexpr.ITs) do it
                IT(it.g, copy(it.k))
            end
    
            ITexpr([ITs..., IT(g, k)])
        elseif mutation_id==2 # interp
            n_terms = size(itexpr.ITs, 1)

            idx1, idx2 = rand(1:n_terms, 2)

            # Vamos pegar a função de um deles, que seja do primeiro
            g = itexpr.ITs[idx1].g

            k1, k2 = itexpr.ITs[idx1].k, itexpr.ITs[idx2].k
    
            # Vamos ver quantos não nulos tem e pegar no máximo max_nonzero_k
            # aleatórios para usar
            nonzero_k = [i for i in 1:nvars if k1[i] != 0 || k2[i]!=0]
            
            nonzero_k = nonzero_k[ randperm(size(nonzero_k, 1))[1:min(size(nonzero_k, 1), max_nonzero_k)] ]
            
            k = zeros(nvars)
            # Usando min e max para controlar os intervalos
            k[nonzero_k] = min.(max.(k1[nonzero_k] .+ k2[nonzero_k], min_expo), max_expo)

            ITs = map(itexpr.ITs) do it
                IT(it.g, copy(it.k))
            end
    
            ITexpr([ITs..., IT(g, k)])

        elseif mutation_id==3 # intern
            n_terms = size(itexpr.ITs, 1)

            idx1, idx2 = rand(1:n_terms, 2)

            # Vamos pegar a função de um deles, que seja do primeiro
            g = itexpr.ITs[idx1].g

            k1, k2 = itexpr.ITs[idx1].k, itexpr.ITs[idx2].k
    
            # Vamos ver quantos não nulos tem e pegar no máximo max_nonzero_k
            # aleatórios para usar
            nonzero_k = [i for i in 1:nvars if k1[i] != 0 || k2[i]!=0]
            
            nonzero_k = nonzero_k[ randperm(size(nonzero_k, 1))[1:min(size(nonzero_k, 1), max_nonzero_k)] ]
            
            k = zeros(nvars)
            # Usando min e max para controlar os intervalos
            k[nonzero_k] = min.(max.(k1[nonzero_k] .- k2[nonzero_k], min_expo), max_expo)

            ITs = map(itexpr.ITs) do it
                IT(it.g, copy(it.k))
            end
    
            ITexpr([ITs..., IT(g, k)])

        elseif mutation_id==4 # TODO: interaction exponents
            n_terms = size(itexpr.ITs, 1)
            idx = rand(1:n_terms)

            itexpr
        else # drop
            n_terms = size(itexpr.ITs, 1)
            idx = rand(1:n_terms)

            ITs = [IT(itexpr.ITs[i].g, copy(itexpr.ITs[i].k))
                for i in (1:idx-1..., idx+1:n_terms...)]
    
            ITexpr(ITs)
        end
    end)
end


function _tournament_selection(competitors, c_fitness)
    
    # TODO: usar argmin
    winner_idx = reduce(
        (i1, i2) -> c_fitness[i1] < c_fitness[i2] ? i1 : i2,
        1:size(competitors.ITexprs, 1),
        init=1
    )

    competitors.ITexprs[winner_idx], c_fitness[winner_idx]
end

# TODO: fazer validação
function _arguments_validation()
    # Checar se o expolim deixa ter ao menos 1 expoente não nulo
    0
end

# TODO: memorizar cálculos?



# TODO: passo de simplificar. Remover termos redundantes

# Function to rollback zip, will be useful with tournament
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

# TODO: usar mais símbolos e nomes significativos e curtos

function ITEA(
    X::Matrix{Float64}, 
    y::Vector{Float64},
    transf_funcs::Vector{Function},
    max_nonzero_k::Int64;
    tourn_size::Int64=3,
    expo_bounds::Tuple{Int64, Int64}  = (-2, 2),
    terms_bounds::Tuple{Int64, Int64} = (1, 5),
    popsize::Int64  = 250,
    gens::Int64     = 250,
    mutationWeights = ProbabilityWeights([.11, .11, .11, .33, .33]),
    adjust_method::String = "ordinary_least_squares_adj")
    
    #TODO: controle verbose

    nvars = size(X, 2)

    # initializing
    pop = _init_population(popsize, nvars, transf_funcs, expo_bounds,
        terms_bounds, max_nonzero_k)
    
    # Adjusting
    if adjust_method in ["ordinary_least_squares_adj", "levenberg_marquardt_adj"]
        pop = ITpop([eval(Symbol(adjust_method))(p, X, y)
            for p in pop.ITexprs])
    end
    
    # TODO: calcular mais estatísticas
    println("\nGer,\t smlstFit,\t avgFit")

    fitnesses = [RMSE(itexpr, X, y) for itexpr in pop.ITexprs]
    for g in 1:gens

        #TODO: usar minimum e maximum no lugar de min e max quando trabalho com vetores
        println("$g,\t $(minimum(fitnesses)),\t $(mean(fitnesses))")

        xmen = _mutate_population(pop, nvars, mutationWeights, transf_funcs,
        expo_bounds, terms_bounds, max_nonzero_k)

        # TODO: otimizar o uso do adjust method
        if adjust_method in ["ordinary_least_squares_adj", "levenberg_marquardt_adj"]
            xmen = ITpop([eval(Symbol(adjust_method))(xm, X, y)
                for xm in xmen.ITexprs])
        end

        #println(size(pop.ITexprs))

        # Including mutated children. Population now has doubled in size.
        pop = ITpop([pop.ITexprs..., xmen.ITexprs...])
        fitnesses = [fitnesses..., [RMSE(itexpr, X, y) for itexpr in xmen.ITexprs]...]
    
        #println(size(pop.ITexprs))
        #println(size(xmen.ITexprs))

        selected = map(1:popsize) do i
            idxs = rand(1:popsize*2, tourn_size)

            _tournament_selection(ITpop(pop.ITexprs[idxs]), fitnesses[idxs])
        end

        individuals, fitnesses = unzip(selected)

        pop = ITpop(individuals)
    end

    pop.ITexprs[argmin(fitnesses)]
end