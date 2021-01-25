include("../util/serialize.jl")

module DownloadMNIST

using GZip 

DATASET_URL = "http://yann.lecun.com/exdb/mnist/"

FILENAMES = Dict(
    "train_img" => "train-images-idx3-ubyte.gz",
    "train_label" => "train-labels-idx1-ubyte.gz",
    "test_img" => "t10k-images-idx3-ubyte.gz",
    "test_label" => "t10k-labels-idx1-ubyte.gz"
)

TRAIN_NUM = 60000
TEST_NUM = 10000
IMG_SIZE = 784

SAVE_FILE = "mnist.jld2"

function download_dataset()
    for (key, value) in FILENAMES
        if isfile(value)
            continue
        end
        download(DATASET_URL * value, value)
    end
end

function load_image(filename::String)::AbstractArray
    f = GZip.open(filename)
    data = read(f)[17:end]
    data_count = convert(Int64, length(data) / IMG_SIZE) 
    return reshape(data, IMG_SIZE, data_count) 
end

function load_label(filename::String)::AbstractArray
    f = GZip.open(filename)
    data = read(f)[9:end]
    return data
end

function convert_data()
    return Dict(
        "train_img" => load_image(FILENAMES["train_img"]),
        "train_label" => load_label(FILENAMES["train_label"]),
        "test_img" => load_image(FILENAMES["test_img"]),
        "test_label" => load_label(FILENAMES["test_label"])
    )
end

function init_mnist()
    download_dataset()
    dataset = convert_data()
    Serialize.dump_data(SAVE_FILE, dataset)
    print("Done!")
end

end
