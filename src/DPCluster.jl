#
# Dirichlet Mixture Models
#
# By Kira Selby
#

#
# This file contains the functions for performing the clustering algorithms
# themselves, using the utilities defined elsewhere. The clustering is done
# according to the algorithms in Neal (2000). To use the functions defined here,
# either pass in a model from one of the types defined in the models folder or
# define your own model in a similar fashion. Then, invoke DPCluster with your
# your data, the model, and a value for the concentration parameter α
#

#
# Clustering over class labels using Algorithm 2 from Neal(2000).
#

function dp_cluster(Y::Array{Float64,1}, model::UnivariateConjugateModel, α::Float64; iters::Int64=5000, burnin::Int64=200, shuffled::Bool=true)
  states = Array{DMMState, 1}(iters-burnin)

  # Initialize the clusters, returning c and phi
  state::DMMState = DMMState(Y,model)

  # Iterate
  for i in 1:iters
    # Iterate through all Y and update
    state = sample_Y(state,model,α)

    # Iterate through all ϕ and update
    for k in keys(state.ϕ)
      state.ϕ[k] = sample_posterior(model,state.Y[k])
    end
    if i > burnin
      states[i-burnin] = state
    end
  end
  if shuffled
    states = shuffle!(states)
  end
  return state
end
function dp_cluster(Y::Array{Float64,2}, model::MultivariateConjugateModel, α::Float64; iters::Int64=5000, burnin::Int64=200, shuffled::Bool=true)
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
      state.ϕ[k] = sample_posterior(model,state.Y[k])
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
  return state
end

#
# Iterate over all data points in the state, drawing a new cluster for each.
# Returns a new state object.
#

function sample_Y(state::DMMState, model::UnivariateConjugateModel, α::Float64)
  N=sum(values(state.n))
  nextstate = DMMState(state.ϕ,state.n)

  for k in keys(state.Y)
    for y in state.Y[k]
      nextstate.n[k] -= 1
      K = collect(keys(nextstate.n))

      q=[pdf_likelihood(model,y,nextstate.ϕ[i])*nextstate.n[i]/(N-1+α) for i in K]
      r=marginal_likelihood(model,y)*α/(N-1+α)
      b= 1/(r+sum(q))
      r *= b
      q *= b

      rd=rand()
      p=r
      if rd < p
        ϕk = sample_posterior(model,y)
        addnew!(nextstate, y, ϕk)
      else
        for i in 1:length(K)
          p += q[i]
          if rd < p
            addto!(nextstate, y, K[i])
            break
          end
        end
      end
    end
  end
  cleanup!(nextstate)
  return nextstate
end

function sample_Y(state::DMMState, model::MultivariateConjugateModel, α::Float64)
  N=sum(values(state.n))
  nextstate = DMMState(state.ϕ,state.n)

  for k in keys(state.Y)
    Yk = state.Y[k]
    for j in 1:size(Yk, 2)
      yj = Yk[:,j:j]
      nextstate.n[k] -= 1
      K = collect(keys(nextstate.n))

      q=[pdf_likelihood(model,yj[:,1],nextstate.ϕ[i])*nextstate.n[i]/(N-1+α) for i in K]
      r=marginal_likelihood(model,yj[:,1])*α/(N-1+α)

      rbase = r
      qbase = q

      b= 1/(r+sum(q))
      r *= b
      q *= b

      rd=rand()
      p=r
      if rd < p
        ϕk = sample_posterior(model,yj[:,1])
        addnew!(nextstate, yj, ϕk)
      else
        for i in 1:length(K)
          p += q[i]
          if rd < p
            addto!(nextstate, yj, K[i])
            break
          end
        end
      end
    end
  end
  cleanup!(nextstate)
  return nextstate
end
