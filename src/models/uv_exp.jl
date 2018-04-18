
struct UnivariateExponentialModel <: UnivariateConjugateModel
  prior::Gamma
end

function pdf_likelihood(model::UnivariateExponentialModel, y::Float64, θ::Tuple{Float64})
  pdf(Exponential(θ[1]), y)
end
function sample_posterior(model::UnivariateExponentialModel, Y::Array{Float64,1})
  (rand(posterior_canon(model.prior,suffstats(Exponential,Y))),)
end
function sample_posterior(model::UnivariateExponentialModel, y::Float64)
  (rand(posterior_canon(model.prior,suffstats(Exponential,[y]))),)
end
function marginal_likelihood(model::UnivariateExponentialModel, y::Float64)
  M=model.prior
  M.α*(M.θ) / (1+y*M.θ)^(M.α+1)
end
