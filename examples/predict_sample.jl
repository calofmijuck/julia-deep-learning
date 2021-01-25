include("../util/activation.jl")
include("../dataset/mnist.jl")

Network = Dict{String,AbstractArray}

function get_data()::Tuple{AbstractArray,AbstractArray}
    (train_img, train_label), (test_img, test_label) = MNIST.load_mnist(normalize=false)
    return (test_img, test_label)
end

function load_pretrained()::Network
    nn = Serialize.load_data("sample_weight.jld2")
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

function predict(nn::Network, x::AbstractArray)::AbstractArray
    return Activation.softmax(forward(nn, x))
end

image, label = get_data()
nn = load_pretrained()

correct, image_length = 0, size(image)[2]

for i in range(1, image_length, step=1)
    idx = argmax(predict(nn, image[:, i]))
    if idx == label[i] + 1
        global correct += 1
    end
end

print("Accuracy: $(correct / image_length)")
