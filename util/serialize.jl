module Serialize

using JLD2
using FileIO

function dump_data(filename::String, data::Any)
    file = File(format"JLD2", filename)
    save(file, "data", data)
end

function load_data(filename::String)
    file = File(format"JLD2", filename)
    return load(file)["data"]
end

end
