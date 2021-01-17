# sigmoid function
function sigmoid(x::Union{Number, AbstractArray})::Union{Number, AbstractArray}
    return 1 ./ (1 .+ exp.(-x))
end

# alias
σ = sigmoid

# ReLU function
function relu(x::Union{Number, AbstractArray})::Union{Number, AbstractArray}
    return max.(0, x)
end

# Leaky ReLU function
function leaky_relu(x::Union{Number, AbstractArray})::Union{Number, AbstractArray}
    return max.(0.01x, x)
end
