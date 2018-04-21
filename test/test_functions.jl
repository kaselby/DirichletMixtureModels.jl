using DirichletMixtureModels
using Distributions
using ConjugatePriors

importall DirichletMixtureModels
importall Distributions
import ConjugatePriors:
    NormalGamma,
    NormalWishart,
    pdf,
    posterior_canon

function pdf_joint(model::ConjugateModel, y, θ)
    pdf(model.prior,θ...)*pdf_likelihood(model, y, θ)
end

function pdf_posterior(model::UnivariateNormalModel, y, θ)
    pdf(posterior_canon(model.prior,suffstats(Normal, [y])), θ...)
end
function pdf_posterior(model::MultivariateNormalModel, y, θ)
    pdf(posterior_canon(model.prior,suffstats(MvNormal, reshape(y, (length(y),1)))), θ...)
end
function pdf_posterior(model::UnivariateNormalKnownSigma, y, θ)
    pdf(posterior_canon(model.prior,suffstats(NormalKnownSigma(model.σ), [y])), θ...)
end
function pdf_posterior(model::UnivariateExponentialModel, y, θ)
    pdf(posterior_canon(model.prior,suffstats(Exponential, [y])), θ...)
end
