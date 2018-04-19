abstract type AbstractMixtureModel end
abstract type ConjugateModel <: AbstractMixtureModel end


"""
    pdf_likelihood(model, y, θ)
Evaluates the probability density function of the likelihood for the model at y,
given parameters θ.
"""
function pdf_likelihood(model::AbstractMixtureModel, y::Union{Float64, Array{Float64}}, θ::Tuple) end
"""
    sample_posterior(model, y)
Generates a sample from the posterior of the model, conditioned on the data
point (s) contained in y. y may be a Float, a 1D array of floats, or a 2D array
of floats, depending on whether the model is univariate or multivariate and
whether it is conditioned on a single observation or multiple observations.
"""
function sample_posterior(model::AbstractMixtureModel, y::Union{Float64, Array{Float64}}) end
"""
    marginal_likelihood(model, y)
Evaluates the likelihood for the model at y, marginalized over all values of the
parameters θ.
"""
function marginal_likelihood(model::AbstractMixtureModel, y::Union{Float64, Array{Float64}}) end
function standard_form(model::AbstractMixtureModel, ϕ::Tuple)
  ϕ
end
"""
    to_string(model, ϕ)
Converts the parameters for a given cluster in the model to a string. This
is an optional method that is used solely for formatting the output of the MCMC.
"""
function to_string(model::AbstractMixtureModel, ϕ::Tuple)
    return string(ϕ)
end


"""
    sample_prior(model)
Replaces sample_posterior for non-conjugate models. Returns a sample from the prior.
"""
function sample_prior(model) end
