

## SampleCache initialization function ##
function samplecache_initialize(self, variables)

    # Get attributes from self
    name = self.name
    path = self.path
    fields_to_save = self.fields_to_save
    num_iterations = self.num_iterations

    # Ensure variables is a dictionary
    variables = Dict(variables)

    # Create H5 file
    fid = h5open(path*name, "w")

    # Create dataset for each field in fields_to_save
    for field in fields_to_save

        # Get value of and type of field
        value = variables[string(field)]
        type = eltype(value)

        # Create full dataset for field
        dataset = zeros(type, num_iterations, size(value)...)

        # Create chunk that stores value of field at a single iteration
        chunk_size = Int[1, size(value)...]
        if length(chunk_size) == 1
            chunk_size = Int[1, 1]
            dataset = reshape(dataset, num_iterations, 1)
        end
        
        # Create dataset for field
        create_dataset(fid, field, dataset, chunk=chunk_size)
    end

    # Create group for MAP
    mapfid = create_group(fid, "MAP")
    for (key, value) in pairs(variables)
        try 
            mapfid[String(key)] = value
        catch
            mapfid[String(key)] = repr(value)
        end
    end

    # Create group for Last Iteration
    mapfid = create_group(fid, "LAST")
    for (key, value) in pairs(variables)
        try 
            mapfid[String(key)] = value
        catch
            mapfid[String(key)] = repr(value)
        end
    end

    # Close H5 file
    close(fid)

    # Return nothing
    return nothing
end

## SampleCache update function ##
function samplecache_update(self, variables, iteration; isMAP=false)
            
    # Get attributes
    path = self.path
    name = self.name

    # Initialize if iteration is 1
    if iteration == 1
        self.initialize(variables)
    end

    # Ensure variables is a dictionary
    variables = Dict(variables)
        
    # Open the H5 file
    fid = h5open(path*name, "r+")

    # find fields to save
    fields_to_save = keys(fid)
    fields_to_save = fields_to_save[lowercase.(fields_to_save) .!== "map"]

    # Save variables
    for field in fields_to_save

        # Get value of field
        value = variables[string(field)]

        # Save value to chunk
        chunk_dims = ndims(fid[field])
        chunk = Any[iteration; fill(:, chunk_dims-1)]
        fid[field][chunk...] = value
    end

    # Save current iteration to last
    delete_object(fid, "LAST")
    create_group(fid, "LAST")
    lastfid = fid["LAST"]
    for (key, value) in pairs(variables)
        try
            lastfid[String(key)] = value
        catch
            lastfid[String(key)] = repr(value)
        end
    end

    # If variables is the current maximum a posteriori (MAP) then save to MAP group
    if isMAP
        delete_object(fid, "MAP")
        create_group(fid, "MAP")
        mapfid = fid["MAP"]
        for (key, value) in pairs(variables)
            try
                mapfid[String(key)] = value
            catch
                mapfid[String(key)] = repr(value)
            end
        end
    end
    
    # Close the H5 file
    close(fid)

    # Return nothing
    return nothing
end

## Get samples after MCMC ##
function samplecache_get(self, field)
    
    # Get attributes
    path = self.path
    name = self.name

    # Open the H5 file
    fid = h5open(path*name, "r")

    # Check for request
    if lowercase(field) == "map"
        # If MAP is requested then get MAP group
        mapfid = fid["MAP"]
        output = Dict([(Symbol(key), read(mapfid[key])) for key in keys(mapfid)])
    elseif lowercase(field) == "last"
        # If Last is requested then get Last group
        lastfid = fid["LAST"]
        output = Dict([(Symbol(key), read(lastfid[key])) for key in keys(lastfid)])
    elseif lowercase(field) == "fid"
        # If fid is requested then return fid
        return fid
    else
        # Otherwise return samples of field
        output = read(fid[field])  
    end

    # Close the H5 file
    close(fid)

    # Return samples
    return output
end


### Sample Cache ###
mutable struct SampleCache
    name::String
    path::String
    num_iterations::Union{Int, Nothing}
    fields_to_save::Union{Array, Nothing}
    initialize::Function
    update::Function
    get::Function
    function SampleCache(name, num_iterations=nothing, fields_to_save=Any[]; path=nothing)

        # Set extension for savename
        if lowercase(splitext(name)[2]) != ".h5"
            name = name * ".h5"
        end
    
        # Set up path
        if path === nothing
            # Get path from environment variable
            if "PATHTOSAMPLECACHE" in keys(ENV)
                path = ENV["PATHTOSAMPLECACHE"]
            else
                path = "./"
            end
        end
        if path[end] != '/'
            path = path * "/"
        end

        # Create self reference
        self = new()

        # Set attributes
        self.name = name
        self.path = path
        self.num_iterations = num_iterations
        self.fields_to_save = fields_to_save
        self.initialize = function (variables)
            samplecache_initialize(self, variables)
        end
        self.update = function (variables, iteration; isMAP=false)
            samplecache_update(self, variables, iteration; isMAP=isMAP)
        end
        self.get = function (field)
            samplecache_get(self, field)
        end

        return self
    end
end
function SampleCache(name; num_iterations=nothing, fields_to_save=[], path=nothing)
    self = SampleCache(name, num_iterations, fields_to_save; path=path)
    return self
end