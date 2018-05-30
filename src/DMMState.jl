#
# Dirichlet Mixture Models
#
# By Kira Selby
#

#
# The code in this file provides utilities for storing and manipulating
# the current state of the Markov Chain for use in performing Gibbs Sampling
# from a Dirichlet Mixture Model (see DPCLuster.jl)
#


#
# Stores the current state of the clusters in the Markov Chain.
#
"""
    DMMState(ϕ, Y, n)
Stores the current state of the Dirichlet Mixture Model Markov Chain. Consists
of three dictionaries, each with the same keys.

`ϕ` maps the cluster index to a tuple of parameters. The form of this tuple may
depend on the model.
`Y` maps the cluster index to a list of datapoints currently associated with that
cluster. It is a 1D array for univariate models, or an dxN matrix for
multivariate models - where d is the dimensionality of the data and N is the
number of data points in the cluster.
`n` maps the cluster index to the number of elements in the cluster.
`DMMState.n[k]` is the same as `length(DMMState.Y[k])` or
`size(DMMState.Y[k], 2)` for the uni- or multi-variate cases respectively.

```julia
DMMState()            # Creates an empty state
DMMState(ϕ, n)        # Initializes a state from an existing state. Does NOT
                      # initialize data points. Used in `sample_Y`.
DMMState(Y, model)    # Randomly initializes a state from a given dataset and
                      # model specification.
```
"""
struct DMMState
  ϕ::Dict{Int64,Tuple}
  Y::Dict{Int64,Array{Int64,1}}
  n::Dict{Int64,Int64}
end

#
# Constructors to create an empty state, build a new state from an existing state,
# or create a new state from unlabelled data
#

function DMMState()
  return DMMState(Dict{Int64,Tuple}(),Dict{Int64,Array{Int64,1}}(),
                    Dict{Int64,Int64}())
end

function DMMState(s::DMMState)
  return DMMState(copy(s.ϕ), Dict{Int64,Array{Int64,1}}(), copy(s.n))
end

function DMMState(data::Array{Float64}, model::ConjugateModel)
  N=size(data)[end]
  ϕ=Dict{Int64,Tuple}()
  Y=Dict{Int64,Array{Int64,1}}()
  n=Dict{Int64,Int64}()
  for i in 1:N
    ϕ[i] = sample_posterior(model,get_data(data,i))
    Y[i] = [i]
    n[i] = 1
  end
  return DMMState(ϕ,Y,n)
end
function DMMState(data::Array{Float64}, model::NonConjugateModel, m::Int64)
  N=size(data)[end]
  ϕ=Dict{Int64,Tuple}()
  Y=Dict{Int64,Array{Int64,1}}()
  n=Dict{Int64,Int64}()
  for i in 1:N
    ϕ[i] = sample_posterior(model,get_data(data,i),m)
    Y[i] = [i]
    n[i] = 1
  end
  return DMMState(ϕ,Y,n)
end

"""
  get_data(Y, I)
Assembles data points from `Y` indexed by `I`, where Y can be a vector or matrix
and I can be a single index or an indexing set.
"""
function get_data(Y, I) end

get_data(Y::Array{Float64, 1}, I::Array{Int64,1})=Y[I]
get_data(Y::Array{Float64,2}, I::Array{Int64,1})=Y[:,I]
get_data(Y::Array{Float64, 1}, i::Int64)=Y[i]
get_data(Y::Array{Float64, 2}, i::Int64)=Y[:,i]


#
# Add new data to the state (when the label is completely unknown)
#

"""
    add!(state, y, ϕ)
Adds a new data point to a `DMMState` object, assuming its cluster parameters are
known, but the label is not.
"""
function add!(state, y, ϕ) end

function add!(state::DMMState, yi::Float64, ϕi::Tuple)
  added=false
  for (k,v) in state.Y
    if isequalϵ(ϕi,state.ϕ[i])
      addto!(state, yi, i)
      added=true
    end
  end
  if added == false
    addnew!(state,yi,ϕi)
  end
end
function add!(state::DMMState, yi::Array{Float64,2}, ϕi::Tuple)
  added=false
  for (k,v) in state.Y
    if isequalϵ(ϕi,state.ϕ[i])
      addto!(state, yi, i)
      added=true
    end
  end
  if added == false
    addnew!(state,yi,ϕi)
  end
end

"""
    addnew!(state, y, ϕ)
Adds a new cluster to a `DMMState` object, with initial data y and parameters ϕ.
Assumes the label is not known.
"""
function addnew!(state, y, ϕ) end

function addnew!(state::DMMState, j::Int64, ϕj::Tuple)
  i = 1
  K = keys(state.n)
  while i in K
    i += 1
  end
  state.Y[i] = [j]
  state.n[i] = 1
  state.ϕ[i] = ϕj
  return i
end

#
# Add new data to the state (when it is known that the data belongs to a new cluster, and the desired label is known)
#

"""
    addnew!(state, j, ϕ, i)
Adds a new cluster to a `DMMState` object, with initial data label `j`, parameters `ϕ`, and
cluster label `i`.
"""
function addnew!(state::DMMState, j::Int64, ϕi::Tuple, i::Int64)
  state.Y[i] = [j]
  state.n[i] = 1
  state.ϕ[i] = ϕi
end

#
# Add new data to the state (when the cluster label is known). ϕ is assumed to already be accurate.
#

"""
    addto!(state, j, i)
Adds a new data point with label `j` to the `i`th existing cluster of a `DMMState` object.
"""
function addto!(state::DMMState, j::Int64, i::Int64)
  @assert i in keys(state.ϕ)
  @assert i in keys(state.n)
  state.n[i]+=1
  if i in keys(state.Y)
    append!(state.Y[i], j)
  else
    state.Y[i] = [j]
  end
end


"""
    cleanup!(state)
Removes all empty clusters from a `DMMState` object. Y is assumed to be accurate.
"""
function cleanup!(state::DMMState)
  for (k,v) in state.n
    if v==0
      delete!(state.n, k)
      delete!(state.ϕ, k)
    end
  end
end


"""
    isequalϵ(a,b,ϵ=1e-6)
Checks if two numbers, arrays, or tuples thereof are equal to within relative
error `ϵ`.
"""
function isequalϵ(a,b,ϵ=1e-6) end

function isequalϵ(a::Number, b::Number, ϵ=1e-6)
  return abs(a-b)/min(a,b) < ϵ
end

#
# Check if two arrays (of same sizes) are equal (within relative error ϵ)
#
function isequalϵ(a::AbstractArray, b::AbstractArray, ϵ=1e-6)
  @assert length(a)==length(b)
  if length(a)==1
    return isequalϵ(a[1], b[1], ϵ)
  else
    for i in length(a)
      if !isequalϵ(a[i],b[i],ϵ)
        return false
      end
    end
    return true
  end
end

#
# Check if two tuples of arrays (of same sizes) are equal (within relative error ϵ)
#
function isequalϵ(a::Tuple, b::Tuple, ϵ=1e-6)
  @assert length(a)==length(b)
  for i in length(a)
    if !isequalϵ(a[i], b[i], ϵ)
      return false
    end
  end
  return true
end


#
# Export the contents of the state
#
function export_state(data::Array{Float64,1}, model::AbstractMixtureModel, s::DMMState)
  N=length(data)
  K=collect(keys(s.n))
  m=length(K)

  labelled_data=zeros(Float64,2,N)
  phi=Array{Array,1}(m)
  n=Array{Int64,1}(m)
  for i in 1:m
    k=K[i]
    phi[i]=collect(standard_form(model, s.ϕ[k]))
    n[i]=s.n[k]
    J=s.Y[k]
    labelled_data[1,J]=i
    labelled_data[2,J]=data[J]
  end
  [transpose(labelled_data), phi, n]
end
function export_state(data::Array{Float64,2}, model::AbstractMixtureModel, s::DMMState)
  d,N=size(data)
  K=collect(keys(s.n))
  m=length(K)

  labelled_data=zeros(Float64,d+1,N)
  phi=Array{Array,1}(m)
  n=Array{Int64,1}(m)
  for i in 1:m
    k=K[i]
    phi[i]=collect(standard_form(model, s.ϕ[k]))
    n[i]=s.n[k]
    J=s.Y[k]
    labelled_data[1,J]=i
    labelled_data[2:end,J]=data[:,J]
  end
  [transpose(labelled_data), phi, n]
end

function export_all(data::Array{Float64}, model::AbstractMixtureModel, s::Array{DMMState,1})
  M=length(s)
  states=Array{Array, 1}(M)
  for j in 1:M
    states[j] = export_state(data,model,s[j])
  end
  states
end
