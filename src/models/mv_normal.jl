
struct MultivariateNormalModel <: MultivariateConjugateModel
  prior::NormalWishart
end

function MultivariateNormalModel(ss::MvNormalStats)
  p=NormalWishart(ss.m, 1e-8, ss.s2/ss.tw, Float64(length(ss.m)))
  MultivariateNormalModel(p)
end
function MultivariateNormalModel(d::Int64)
  p=NormalWishart(zeros(d), 1.0, eye(d), d*1.0)
  MultivariateNormalModel(p)
end
function MultivariateNormalModel()
  d=2
  p=NormalWishart(zeros(d), 1.0, eye(d), d*1.0)
  MultivariateNormalModel(p)
end

function pdf_likelihood(model::MultivariateNormalModel, y::Array{Float64,1}, θ::Tuple{Array{Float64,1},Array{Float64,2}})
  pdf(MvNormalCanon(θ[2]*θ[1], θ[2]), y)
end
function sample_posterior(model::MultivariateNormalModel, Y::Array{Float64,2})
  p=posterior_canon(model.prior,suffstats(MvNormal,Y))
  rand(p)
end
function sample_posterior(model::MultivariateNormalModel, y::Array{Float64,1})
  p=posterior_canon(model.prior,suffstats(MvNormal,reshape(y,(length(y),1))))
  rand(p)
end
function marginal_likelihood(model::MultivariateNormalModel, y::Array{Float64,1})
  d=length(y)
  p = model.prior
  mu0 = p.mu
  kappa0 = p.kappa
  TC0 = p.Tchol
  nu0 = p.nu

  kappa = kappa0 + 1
  nu = nu0 + 1
  mu = (kappa0.*mu0 + y) ./ kappa
  z = p.zeromean ? y : y - mu0
  Lam = PDMat(Symmetric(inv(inv(TC0) + kappa0/kappa*(z*z'))))

  exp(-d/2*log(π) + logmvgamma(d,nu/2) - logmvgamma(d,nu0/2) + nu0/2*logdet(TC0) - nu/2*logdet(Lam) + d/2 * (log(kappa0) - log(kappa)))
end
function standard_form(model::MultivariateNormalModel, ϕ::Tuple{Float64, Float64})
  (ϕ[1], inv(ϕ[2]))
end
function to_string(model::MultivariateNormalModel, ϕ::Tuple{Array{Float64,1}, Array{Float64,2}})
  "Mean: $(ϕ[1]), Covariance Matrix: $(inv(ϕ[2]))"
end
