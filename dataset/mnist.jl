include("download_mnist.jl")

module MNIST

using GZip

using Main.DownloadMNIST
using Main.Serialize

NUM_LABELS = 10

function convert_one_hot_label(labels::AbstractArray)::AbstractArray
    one_hot_label = zeros(NUM_LABELS, length(labels))
    for (idx, val) in enumerate(labels)
        # Note that array indices are 1-based
        one_hot_label[val + 1, idx] = 1
    end
    return one_hot_label
end

function load_mnist(;normalize::Bool=true, flatten::Bool=true, one_hot_label::Bool=false)
    if isfile(DownloadMNIST.SAVE_FILE) == false
        DownloadMNIST.init_mnist()
    end

    dataset = convert(Dict{String, Any}, Serialize.load_data(DownloadMNIST.SAVE_FILE))

    if normalize
        for key in ("train_img", "test_img")
            dataset[key] = convert.(Float32, dataset[key]) ./ 255.0
        end
    end

    if one_hot_label
        dataset["train_label"] = convert_one_hot_label(dataset["train_label"])
        dataset["test_label"] = convert_one_hot_label(dataset["test_label"])
    end

    if flatten == false
        for key in ("train_img", "test_img")
            data_count = convert(Int64, length(dataset[key]) / DownloadMNIST.IMG_SIZE)
            dataset[key] = reshape(dataset[key], data_count, 1, 28, 28)
        end
    end

    return (
        (dataset["train_img"], dataset["train_label"]),
        (dataset["test_img"], dataset["test_label"])
    )
end

end
