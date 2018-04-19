
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
function dp_cluster(Y::Array{Float64}, model::ConjugateModel, α::Float64; iters::Int64=5000, burnin::Int64=200, shuffled::Bool=true)
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
  export_states(model, states)
end

function dp_cluster(Y::Array{Float64}, model::NonConjugateModel, α::Float64; m_prior::Int64=3, m_post::Int64=3, iters::Int64=5000, burnin::Int64=200, shuffled::Bool=true)
  # Initialize the array of states
  states = Array{DMMState, 1}(iters-burnin)

  # Initialize the clusters, returning c and phi
  state::DMMState = DMMState(Y,model)

  # Iterate
  for i in 1:iters
    # Iterate through all Y and update
    state = sample_Y(state, model, α, m_prior)

    # Iterate through all ϕ and update
    for k in keys(state.ϕ)
      state.ϕ[k] = sample_posterior(model,get_data(state.data,state.Y[k]), m_post)
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
  export_states(model, states)
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
  nextstate = DMMState(state)
  N=sum(values(state.n))
  K = collect(keys(nextstate.n))

  for k in keys(state.Y)
    for j in state.Y[k]
      nextstate.n[k] -= 1
      _sample_y!(j, nextstate, model, α, N, K)
    end
  end
  cleanup!(nextstate)
  return nextstate
end


function sample_Y(state::DMMState, model::NonConjugateModel, α::Float64, m::Int64)
  N=sum(values(state.n))
  K=collect(keys(state.n))
  L = length(K)
  nextstate = DMMState(state)

  aux=Array{Tuple}(m)
  for (k,v) in state.n
      for j=1:v
          nextstate.n[k] -= 1
          for i=1:m
              if (v==1) & (i==1)
                  aux[1] = s.ϕ[k]
              else
                  aux[i] = sample_prior(model)
              end
          end
          _sample_y!(nextstate, state.Y[k][j], aux, model, α, m, N, K, L)
      end
  end
  cleanup!(nextstate)
  return nextstate
end

function _sample_y!(j::Int64, nextstate::DMMState, model::ConjugateModel, α::Float64, N::Int64, K::Array{Int64,1})
  yj = get_data(nextstate.data, j)

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

function _sample_y!(nextstate::DMMState, j::Int64, aux::Array{Tuple}, model::NonConjugateModel, α::Float64, m::Int64, N::Int64, K::Array{Int64,1}, L::Int64)
    y=get_data(state.data, j)

    # Get likelihood probabilities
    q_prev=[pdf_likelihood(model, y, nextstate.ϕ[k])*nextstate.n[k]/(N-1+α) for k in K]
    q_aux=[pdf_likelihood(model, y, θ)*α/m/(N-1+α) for θ in aux]
    b = sum(q_prev) + sum(q_aux)
    q_prev = q_prev/b
    q_aux = q_aux/b

    rd=rand()
    p=0
    for i in 1:(L+m)
        if i <= L
            p += q_prev[i]
            if rd < p
              addto!(nextstate, j, K[i])
              break
            end
        else
            p += q_aux[i-L]
            if rd < p
              addnew!(nextstate, j, aux[i-L])
              break
            end
        end
    end
end
