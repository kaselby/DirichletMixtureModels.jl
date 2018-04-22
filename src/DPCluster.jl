
"""
    dp_cluster(Y, model, α, iters=5000, burnin=200, shuffled=true)
    dp_cluster(Y, model, α, iters=5000, burnin=200, shuffled=true, m_prior=2, m_post=5)
In the conjugate case, performs `iters` iterations of Algorithm 2 from Neal(2000)
to generate possible clusters for the data in `Y`, using the model in `model`,
with concentration parameter `alpha`.
In the 1D case, Y is assumed to be a 1D array of floats. In the 2D case,
`Y` is assumed to be a dxN array of floats, where the data is d-dimensional
and N is the number of datapoints.
Returns a 1D array of DMMState objects. This includes the states after each
post-burnin iteration, with the default being a burnin of 500. By default, this
array is shuffled so that it may be used to approximate I.I.D draws from the
posterior.
"""
function dp_cluster(Y, model, α, iters=5000,burnin=500, shuffled=true) end

"""
  sample_Y(state, data, model, α)
  sample_Y(state, data, model, α, m)
Iterates through each data point in the given `DMMState` object, drawing a new
cluster for each.
Returns a new state object. Used in dp_cluster to perform the Gibbs update on Y.
"""
function sample_Y(state, data, model, α) end

"""
  sample_ϕ(state, data, model, α, m)
Iterates through each cluster in the given `DMMState` object, drawing new parameters
for each.
Returns a new state object. Used in dp_cluster in the non-conjugate case to perform
the Gibbs update on ϕ.
"""
function sample_ϕ(state, data, model, α) end


function dp_cluster(Y::Array{Float64}, model::ConjugateModel, α::Float64; iters::Int64=5000, burnin::Int64=500, shuffled::Bool=true)
  # Initialize the array of states
  states = Array{DMMState, 1}(iters-burnin)

  # Initialize the clusters, returning c and phi
  state::DMMState = DMMState(Y,model)

  # Iterate
  for i in 1:iters
    # Iterate through all Y and update
    state = sample_Y(state,Y,model,α)

    # Iterate through all ϕ and update
    for k in keys(state.ϕ)
      state.ϕ[k] = sample_posterior(model,get_data(Y,state.Y[k]))
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
  states
end


function sample_Y(state::DMMState, data::Array{Float64}, model::ConjugateModel, α::Float64)
  nextstate = DMMState(state)
  N = sum(values(state.n))
  for k in keys(state.Y)
    for j in state.Y[k]
      nextstate.n[k] -= 1
      yj = get_data(data, j)
      _sample_y!(nextstate, yj, j, model, α, N)
    end
  end
  cleanup!(nextstate)
  return nextstate
end
function _sample_y!(nextstate::DMMState, yj::Union{Float64, Array{Float64,1}}, j::Int64, model::ConjugateModel, α::Float64, N::Int64)
  K= collect(keys(nextstate.n))
  q=[pdf_likelihood(model,yj,nextstate.ϕ[i])*nextstate.n[i]/(N-1+α) for i in K]
  r=marginal_likelihood(model,yj)*α/(N-1+α)
  b= r+sum(q)
  r /= b
  q /= b

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


#
# Non-Conjugate case
#

function dp_cluster(Y::Array{Float64}, model::NonConjugateModel, α::Float64; m_prior::Int64=2, m_post::Int64=5, iters::Int64=5000, burnin::Int64=200, shuffled::Bool=true)
  # Initialize the array of states
  states = Array{DMMState, 1}(iters-burnin)

  # Initialize the clusters, returning c and phi
  state::DMMState = DMMState(Y,model,m_post)

  # Iterate
  for i in 1:iters
    # Iterate through all Y and update
    state = sample_Y(state, Y, model, α, m_prior)

    # Iterate through all ϕ and update
    sample_ϕ!(state, Y, model, m_post)

    # Add to the list of states
    if i > burnin
      states[i-burnin] = state
    end
  end
  # Shuffle the states so they may be treated as approximately IID samples
  if shuffled
    states = shuffle!(states)
  end
  states
end

function sample_Y(state::DMMState, data::Array{Float64}, model::NonConjugateModel, α::Float64, m::Int64)
  N=sum(values(state.n))
  nextstate = DMMState(state)

  aux=Array{Tuple}(m)
  for k in keys(state.Y)
      for j in state.Y[k]
          nextstate.n[k] -= 1
          for i=1:m
              if (nextstate.n[k]==1) & (i==1)
                  aux[1] = state.ϕ[k]
              else
                  aux[i] = sample_prior(model)
              end
          end
          yj=get_data(data,j)
          _sample_y!(nextstate, yj, j, aux, model, α, m, N)
      end
  end
  cleanup!(nextstate)
  return nextstate
end
function _sample_y!(nextstate::DMMState, yj::Union{Float64, Array{Float64,1}}, j::Int64, aux::Array{Tuple}, model::NonConjugateModel, α::Float64, m::Int64, N::Int64)
    K=collect(keys(nextstate.n))
    L=length(K)

    # Get likelihood probabilities
    q_prev=[pdf_likelihood(model, yj, nextstate.ϕ[k])*nextstate.n[k]/(N-1+α) for k in K]
    q_aux=[pdf_likelihood(model, yj, θ)*α/m/(N-1+α) for θ in aux]
    b = sum(q_prev) + sum(q_aux)
    q_prev = q_prev/b
    q_aux = q_aux/b

    rd=rand()
    p=0.0
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
function sample_ϕ!(state::DMMState, data::Array{Float64}, model::NonConjugateModel, m::Int64)
  for k in keys(state.ϕ)
    Y=get_data(data,state.Y[k])
    aux=Array{Tuple}(m)
    aux[1]=state.ϕ[k]
    for i=2:m
        aux[i] = sample_prior(model)
    end
    state.ϕ[k]=_sample_ϕ(Y,aux,model,m)
  end
end
function _sample_ϕ(Y::Array{Float64}, aux::Array{Tuple}, model::NonConjugateModel, m::Int64)
  q_aux=[prod(pdf_likelihood(model, Y, θ)) for θ in aux]
  b=sum(q_aux)

  q_aux/=b

  rd=rand()
  p=0
  for i in 1:m
      p += q_aux[i]
      if rd < p
          return aux[i]
      end
  end
  println("This shouldn't happen.")
  println(q_aux)
end
