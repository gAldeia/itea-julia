# TODO: more Metrics, possibility to choose metric when calling ITEA

function RMSE(itexpr, X, y)
    try
        rmse = sqrt( mean( (evaluate(itexpr, X) .- y).^2 ) )

        isfinite(rmse) ? rmse : Inf
    catch DomainError
        return Inf
    end
end