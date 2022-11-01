

## SampleCache initialization function ##
function samplecache_initialize(self, variables)

    # Get attributes from self
    name = self.name
    path = self.path
    fields_to_save = self.fields_to_save
    fields_to_average = self.fields_to_average
    num_iterations = self.num_iterations

    # Ensure variables is a dictionary
    variables = Dict{Symbol, Any}([(Symbol(key), val) for (key, val) in pairs(variables)])

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

    # Create mean group for each field in fields_to_average
    meanfid = create_group(fid, "MEAN")
    for field in fields_to_average

        # Get value of and type of field
        value = variables[string(field)]
        
        # Create mean dataset for field
        create_dataset(meanfid, field, value)
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

    # Create group for MAP
    lastfid = create_group(fid, "LAST")
    for (key, value) in pairs(variables)
        try 
            lastfid[String(key)] = value
        catch
            lastfid[String(key)] = repr(value)
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
    fields_to_save = self.fields_to_save
    fields_to_average = self.fields_to_average

    # Initialize if iteration is 1
    if iteration == 1
        self.initialize(variables)
    end

    # Ensure variables is a dictionary
    variables = Dict{Symbol, Any}([(Symbol(key), val) for (key, val) in pairs(variables)])
        
    # Open the H5 file
    fid = h5open(path*name, "r+")

    # Save variables
    for field in fields_to_save

        # Get value of field
        value = variables[string(field)]

        # Save value to chunk
        chunk_dims = ndims(fid[field])
        chunk = Any[iteration; fill(:, chunk_dims-1)]
        fid[field][chunk...] = value
    end

    # Save running mean
    meanfid = fid["MEAN"]
    for field in fields_to_average

        # Get value of field
        value = variables[string(field)]
        oldmean = meanfid[field]

        # Update mean and variance
        delete_object(meanfid, field)
        meanfid[field] = ((iteration-1) .* oldmean .+ value) ./ iteration
    end

    # Save current iteration to last
    varfid = fid["LAST"]
    for (key, value) in pairs(variables)
        delete_object(varfid, String(key))
        try
            varfid[String(key)] = value
        catch
            varfid[String(key)] = repr(value)
        end
    end

    # If variables is the current maximum a posteriori (MAP) then save to MAP group
    if isMAP
        varfid = fid["MAP"]
        for (key, value) in pairs(variables)
            delete_object(varfid, String(key))
            try
                varfid[String(key)] = value
            catch
                varfid[String(key)] = repr(value)
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
        varfid = fid["MAP"]
        output = Dict{Symbol, Any}([(Symbol(key), read(varfid[key])) for key in keys(varfid)])
    elseif lowercase(field) == "last"
        # If Last is requested then get Last group
        varfid = fid["LAST"]
        output = Dict{Symbol, Any}([(Symbol(key), read(varfid[key])) for key in keys(varfid)])
    elseif lowercase(field) == "mean"
        # If Last is requested then get Last group
        varfid = fid["MEAN"]
        output = Dict{Symbol, Any}([(Symbol(key), read(varfid[key])) for key in keys(varfid)])
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
    fields_to_average::Union{Array, Nothing}
    initialize::Function
    update::Function
    get::Function
    function SampleCache(name, num_iterations, fields_to_save, fields_to_average, path)

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
        self.fields_to_average = fields_to_average
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
function SampleCache(name; num_iterations=nothing, fields_to_save=[], fields_to_average=[], path=nothing)
    self = SampleCache(name, num_iterations, fields_to_save, fields_to_average, path)
    return self
end