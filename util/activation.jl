module Activation

# sigmoid function
sigmoid(x::Number)::Number = 1 / (1 + exp(-x))

sigmoid(x::AbstractArray)::AbstractArray = 1 ./ (1 .+ exp.(-x))

# alias
σ = sigmoid

# ReLU function
relu(x::Number)::Number = max(0, x)

relu(x::AbstractArray)::AbstractArray = max.(0, x)

# Leaky ReLU function
leaky_relu(x::Number)::Number = max(0.01x, x)

leaky_relu(x::AbstractArray)::AbstractArray = max.(0.01x, x)

# Softmax
function softmax(x::AbstractArray)::AbstractArray
    x = x .- maximum(x)
    return exp.(x) ./ sum(exp.(x))
end

end
