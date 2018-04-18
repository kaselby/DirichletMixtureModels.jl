"""
    UnivariateNormalKnownSigma(prior, σ)
This is the base type for the model with `NormalKnownSigma` likelihood and
`Normal` prior.

```julia
UnivariateNormalKnownSigma(prior, σ)    # Creates a model with a given Normal
                                        # prior and fixed variance σ
UnivariateNormalKnownSigma(μ0, σ0, σ)   # Creates a model with prior mean μ0,
                                        # prior variance σ0 and fixed likelihood
                                        # variance σ
UnivariateNormalKnownSigma(ss, σ)       # Creates a model with prior hyper-
                                        # -parameters inferred from the data
                                        # using a NormalStats object
```
"""
struct UnivariateNormalKnownSigma <: UnivariateConjugateModel
  prior::Normal
  σ::Float64
end

function UnivariateNormalKnownSigma(μ0::Float64, σ0::Float64, σ::Float64)
  UnivariateNormalKnownSigma(Normal(μ0, σ0), σ)
end
function UnivariateNormalKnownSigma(ss::NormalStats, σ::Float64)
  p=Normal(ss.m,ss.s2/ss.tw)
  UnivariateNormalKnownSigma(p, σ)
end

function pdf_likelihood(model::UnivariateNormalKnownSigma, y::Float64, θ::Tuple{Float64})
  pdf(Normal(θ..., model.σ), y)
end
function sample_posterior(model::UnivariateNormalKnownSigma, Y::Array{Float64,1})
  p=posterior_canon(model.prior,suffstats(NormalKnownSigma(model.σ),Y))
  (rand(p),)
end
function sample_posterior(model::UnivariateNormalKnownSigma, y::Float64)
  p=posterior_canon(model.prior,suffstats(NormalKnownSigma(model.σ),[y]))
  (rand(p),)
end
function marginal_likelihood(model::UnivariateNormalKnownSigma, y::Float64)
  tau = (model.σ^2 + model.prior.σ^2)^(1/2)
  pdf(Normal(model.prior.μ, tau), y)
end
function to_string(model::UnivariateNormalKnownSigma, ϕ::Tuple{Float64})
  "Mean: $(ϕ[1]), Variance: $(model.σ)"
end

#
# Utility functions for clustering with univariate normal likelihood (mean and precision unknown)
#
"""
    UnivariateNormalModel(prior)
This is the base type for the model with `Normal` likelihood and `NormalGamma`
prior. This is the most commonly used 1D model.

```julia
UnivariateNormalModel(prior)    # Creates a model with a given NormalGamma prior
UnivariateNormalModel(ss)       # Creates a model with prior mean μ0,
                                  # prior variance σ0 and fixed likelihood
                                  # variance σ
UnivariateNormalModel()           # Creates a model with prior hyper-
                                  # -parameters inferred from the data
                                  # using a NormalStats object
```
"""
struct UnivariateNormalModel <: UnivariateConjugateModel
  prior::NormalGamma
end

function UnivariateNormalModel(μ0::Float64, n0::Float64, α0::Float64, β0::Float64)
  UnivariateNormalModel(NormalGamma(μ0, n0, α0, β0))
end
function UnivariateNormalModel(ss::NormalStats)
  p=NormalGamma(ss.m,1e-8,2.0,0.5)
  UnivariateNormalModel(p)
end
function UnivariateNormalModel()
  p=NormalGamma(0.0,1e-8,2.0,0.5)
  UnivariateNormalModel(p)
end

function pdf_likelihood(model::UnivariateNormalModel, y::Float64, θ::Tuple{Float64,Float64})
  pdf(NormalCanon(θ[2]*θ[1], θ[2]), y)
end
function sample_posterior(model::UnivariateNormalModel, Y::Array{Float64,1})
  p=posterior_canon(model.prior,suffstats(Normal,Y))
  rand(p)
end
function sample_posterior(model::UnivariateNormalModel, y::Float64)
  p=posterior_canon(model.prior,suffstats(Normal,[y]))
  rand(p)
end
function marginal_likelihood(model::UnivariateNormalModel, y::Float64)
  p=model.prior
  gamma(p.shape+1/2)/gamma(p.shape) * sqrt(p.nu/(p.nu+1)) * 1/sqrt(2*π) * p.rate^p.shape /
    (p.rate+p.nu/2/(p.nu+1)*(y-p.mu)^2)^(p.shape+1/2)
end
function to_string(model::UnivariateNormalModel, ϕ::Tuple{Float64, Float64})
  "Mean: $(ϕ[1]), Variance: $(1/sqrt(ϕ[2]))"
end
