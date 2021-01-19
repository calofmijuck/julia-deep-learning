include("../util/activation.jl")

Network = Dict{String, AbstractArray}

function init_network()::Network
    nn = Dict()
    nn["W1"] = [0.1 0.2; 0.3 0.4; 0.5 0.6]
    nn["W2"] = [0.1 0.2 0.3; 0.4 0.5 0.6]
    nn["W3"] = [0.1 0.2; 0.3 0.4]
    
    nn["b1"] = [0.1, 0.2, 0.3]
    nn["b2"] = [0.1, 0.2]
    nn["b3"] = [0.1, 0.2]

    return nn
end

function forward(nn::Network, x::AbstractArray)::AbstractArray
    W1, W2, W3 = nn["W1"], nn["W2"], nn["W3"]
    b1, b2, b3 = nn["b1"], nn["b2"], nn["b3"]

    z1 = Activation.sigmoid(W1 * x + b1)
    z2 = Activation.sigmoid(W2 * z1 + b2)
    z3 = W3 * z2 + b3
    
    return z3
end

nn = init_network()
x = [1.0, 0.5]
y = forward(nn, x)
print(y)
