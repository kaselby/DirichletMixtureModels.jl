
"""
    dp_cluster(Y, model, α, iters=5000, burnin=200, shuffled=true)
Performs `iters` iterations of Algorithm 2 from Neal(2000) to generate possible
clusters for the data in `Y`, using the model in `model`, with concentration
parameter `alpha`. In the 1D case, Y is assumed to be a 1D array of floats. In
the 2D case, `Y` is assumed to be a dxN array of floats, where the data is
d-dimensional and N is the number of datapoints.
Returns a 1D array of DMMState objects. This includes the states after each
post-burnin iteration, with the default being a burnin of 200. By default, this
array is shuffled so that it may be used to approximate I.I.D draws from the
posterior.
To see a formatted summary of all the clusters in a given state, call
`summarize(model, state)`, where state is any entry from the outputted list.
"""
function dp_cluster(Y, model, α; iters=5000, burnin=200, shuffled=true) end

# Methods (to enforce proper typing)
function dp_cluster(Y::Array{Float64,1}, model::UnivariateConjugateModel, α::Float64; iters::Int64=5000, burnin::Int64=200, shuffled::Bool=true)
  _dp_cluster(Y, model, α, iters, burnin, shuffled)
end
function dp_cluster(Y::Array{Float64,2}, model::MultivariateConjugateModel, α::Float64; iters::Int64=5000, burnin::Int64=200, shuffled::Bool=true)
  _dp_cluster(Y, model, α, iters, burnin, shuffled)
end

function _dp_cluster(Y::Array{Float64}, model::ConjugateModel, α::Float64, iters::Int64, burnin::Int64, shuffled::Bool)
  # Initialize the array of states
  states = Array{DMMState, 1}(iters-burnin)

  # Initialize the clusters, returning c and phi
  state::DMMState = DMMState(Y,model)

  # Iterate
  for i in 1:iters
    # Iterate through all Y and update
    state = sample_Y(state,model,α)

    # Iterate through all ϕ and update
    for k in keys(state.ϕ)
      state.ϕ[k] = sample_posterior(model,get_data(state.data,state.Y[k]))
    end

    # Add to the list of states
    if i > burnin
      states[i-burnin] = state
    end
  end
  # Shuffle the states so they may be treated as approximately IID samples
  if shuffled
    states = shuffle!(states)
  end
  return states
end

#
# Iterate over all data points in the state, drawing a new cluster for each.
# Returns a new state object.
#

"""
  sample_Y(state, model, α)
Iterates through each data point in the given `DMMState` object, drawing a new
cluster for each.
Returns a new state object.
"""
function sample_Y(state::DMMState, model::ConjugateModel, α::Float64)
  N=sum(values(state.n))
  nextstate = DMMState(state)

  for k in keys(state.Y)
    for j in state.Y[k]
      yj = get_data(state.data, j)
      nextstate.n[k] -= 1
      K = collect(keys(nextstate.n))

      q=[pdf_likelihood(model,yj,nextstate.ϕ[i])*nextstate.n[i]/(N-1+α) for i in K]
      r=marginal_likelihood(model,yj)*α/(N-1+α)
      b= 1/(r+sum(q))
      r *= b
      q *= b

      rd=rand()
      p=r
      if rd < p
        ϕk = sample_posterior(model,yj)
        addnew!(nextstate, j, ϕk)
      else
        for i in 1:length(K)
          p += q[i]
          if rd < p
            addto!(nextstate, j, K[i])
            break
          end
        end
      end
    end
  end
  cleanup!(nextstate)
  return nextstate
end
