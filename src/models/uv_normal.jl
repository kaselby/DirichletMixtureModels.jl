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
struct UnivariateNormalKnownSigma <: ConjugateModel
  prior::Normal
  σ::Float64
end

function UnivariateNormalKnownSigma(μ0::Float64, σ0::Float64, σ::Float64)
  UnivariateNormalKnownSigma(Normal(μ0, σ0), σ)
end
function UnivariateNormalKnownSigma(Y::Array{Float64,1}, σ::Float64)
  ss=suffstats(Normal, Y)
  UnivariateNormalKnownSigma(ss, σ)
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
function standard_form(model::UnivariateNormalKnownSigma, ϕ::Tuple{Float64})
  (ϕ[1], model.σ)
end
function parameter_names(model::UnivariateNormalKnownSigma)
  ("Mean", "Stdev")
end

#
# Utility functions for clustering with univariate normal likelihood (mean and precision unknown)
#
"""
    UnivariateNormalModel(prior)
This is the base type for the model with `Normal` likelihood and `NormalGamma`
prior. This is the most commonly used 1D model.
Note that the `NormalGamma` distribution used in this package uses the shape/
rate parametrization.

```julia
UnivariateNormalModel(prior)          # Creates a model with a given NormalGamma prior.
UnivariateNormalModel(μ0, n0, α0, β0) # Creates a model with the given hyperparameters
                                      # (using shape/rate parametrization)
UnivariateNormalModel(ss)             # Creates a model with prior mean inferred from the
                                      # data using a NormalStats object and default values elsewhere.
UnivariateNormalModel()               # Creates a model with default hyperparameters
                                      # (0, 1e-8, 2, 0.5)
```
"""
struct UnivariateNormalModel <: ConjugateModel
  prior::NormalGamma
end

function UnivariateNormalModel(μ0::Float64, n0::Float64, α0::Float64, β0::Float64)
  UnivariateNormalModel(NormalGamma(μ0, n0, α0, β0))
end
function UnivariateNormalModel(Y::Array{Float64,1})
  ss=suffstats(Normal, Y)
  UnivariateNormalModel(ss)
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
function standard_form(model::UnivariateNormalModel, ϕ::Tuple{Float64,Float64})
  (ϕ[1], 1/sqrt(ϕ[2]))
end
function parameter_names(model::UnivariateNormalModel)
  ("Mean", "Stdev")
end
