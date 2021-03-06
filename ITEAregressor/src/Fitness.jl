# Different regression metrics and a fitness function that implements 
# memoization with the Least Recently Used Cache (LRU)


MAE(pred, y)  = mean( abs.(pred .- y) )

MSE(pred, y)  = mean( (pred .- y).^2 )

NMSE(pred, y) = MSE(pred, y) ./ var(y)

RMSE(pred, y) = sqrt( MSE(pred, y) )

# Negative, since evolution optimizes to minimize the metric
R2neg(pred, y)   = -1(1 - sum((pred .- y).^2) / sum((pred .- mean(y)).^2))


# Methods that are not exported explicitly (memoization can contaminate 
# user results if forgets to clean cache for different data sets)

const _fitness_memoization = LRU{UInt64, Float64}(maxsize = ITEA_CACHE_SIZE)


function fitness(
    itexpr::ITexpr, X::Array{T, 2}, y::Array{T, 1}; metric::String="RMSE"
    ) where {T<:Number}

    metric_error = get!(_fitness_memoization, hash(itexpr)) do
        try
            eval(Symbol(metric))(evaluate(itexpr, X), y)
        catch DomainError
            return Inf
        end
    end
            
    isfinite(metric_error) ? metric_error : Inf
end


function clear_fitness_memoize()

    empty!(_fitness_memoization)
end