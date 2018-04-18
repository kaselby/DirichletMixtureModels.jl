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

struct DMMState
  ϕ::Dict{Int64,Tuple}
  Y::Dict{Int64,AbstractArray{Float64}}
  n::Dict{Int64,Int64}
end

#
# Constructors to create an empty state, build a new state from an existing state,
# or create a new state from unlabelled data
#

function DMMState()
  return DMMState(Dict{Int64,Tuple}(),Dict{Int64,Array{Float64}}(),
                    Dict{Int64,Int64}())
end

function DMMState(ϕ::Dict{Int64,Tuple}, n::Dict{Int64,Int64})
  return DMMState(ϕ,Dict{Int64,Array{Float64}}(), n)
end

function DMMState(Y::Array{Float64,1}, model::UnivariateConjugateModel)
  N=length(Y)
  ϕ=Dict{Int64,Tuple}()
  Ynew=Dict{Int64,Array{Float64,1}}()
  n=Dict{Int64,Int64}()
  for i in 1:N
    ϕ[i] = sample_posterior(model,Y[i])
    Ynew[i] = [Y[i]]
    n[i] = 1
  end
  return DMMState(ϕ,Ynew,n)
end
function DMMState(Y::Array{Float64,2}, model::MultivariateConjugateModel)
  d,N=size(Y)
  ϕ=Dict{Int64,Tuple{AbstractVector,AbstractMatrix}}()
  Ynew=Dict{Int64,Array{Float64,2}}()
  n=Dict{Int64,Int64}()
  for i in 1:N
    ϕ[i] = sample_posterior(model,Y[:,i:i])
    Ynew[i] = Y[:, i:i]
    n[i] = 1
  end
  return DMMState(ϕ,Ynew,n)
end

#
# Add new data to the state (when the label is completely unknown)
#

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

function addnew!(state::DMMState, yi::Float64, ϕi::Tuple)
  i = 1
  K = keys(state.n)
  while i in K
    i += 1
  end
  state.Y[i] = [yi]
  state.n[i] = 1
  state.ϕ[i] = ϕi
  return i
end
function addnew!(state::DMMState, yi::Array{Float64,2}, ϕi::Tuple)
  i = 1
  K = keys(state.n)
  while i in K
    i += 1
  end
  state.Y[i] = yi
  state.n[i] = 1
  state.ϕ[i] = ϕi
  return i
end

#
# Add new data to the state (when it is known that the data belongs to a new cluster, and the desired label is known)
#

function addnew!(state::DMMState, yi::Float64, ϕi::Tuple, i::Int64)
  state.Y[i] = [yi]
  state.n[i] = 1
  state.ϕ[i] = ϕi
end
function addnew!(state::DMMState, yi::Array{Float64,2}, ϕi::Tuple, i::Int64)
  state.Y[i] = yi
  state.n[i] = 1
  state.ϕ[i] = ϕi
end

#
# Add new data to the state (when the cluster label is known). ϕ is assumed to already be accurate.
#

function addto!(state::DMMState, yi::Float64, i::Int64)
  @assert i in keys(state.ϕ)
  @assert i in keys(state.n)
  state.n[i]+=1
  if i in keys(state.Y)
    append!(state.Y[i], yi)
  else
    state.Y[i] = [yi]
  end
end
function addto!(state::DMMState, yi::Array{Float64, 2}, i::Int64)
  @assert i in keys(state.ϕ)
  @assert i in keys(state.n)
  state.n[i]+=1
  if i in keys(state.Y)
    state.Y[i] = [state.Y[i] yi]
  else
    state.Y[i] = yi
  end
end

#
# Clean up state by removing empty clusters. Y is assumed to already be accurate.
#
function cleanup!(state::DMMState)
  for (k,v) in state.n
    if v==0
      delete!(state.n, k)
      delete!(state.ϕ, k)
    end
  end
end

#
# Check if two numbers are equal (within relative error ϵ)
#
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
