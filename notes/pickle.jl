using PyCall

@pyimport pickle

function dump(filename, obj)
    out = open(filename, "w")
    pickle.dump(obj, out)
    close(out)
end

function load(filename)
    pyopen = pybuiltin("open")
    f = pyopen(filename, "rb")
    return pickle.load(f)
end
