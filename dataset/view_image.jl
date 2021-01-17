include("mnist.jl")

using Main.MNIST
using PyCall
using LinearAlgebra

(train_img, train_label), (test_img, test_label) = MNIST.load_mnist(normalize=false)

Image = pyimport("PIL.Image")

function get_image(k::Int64)::AbstractArray
    return train_img[:, k]
end

function view_image(k::Int64)
    pil = Image.fromarray(LinearAlgebra.transpose(reshape(get_image(k), 28, 28)))
    pil.show()
end

view_image(1)

# TODO: takes to long! optimize!
