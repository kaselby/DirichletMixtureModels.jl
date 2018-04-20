#
#   For use with R package
#

struct GeneralConjugateModel <: ConjugateModel
    pdf_likelihood::Function
    sample_posterior::Function
    marginal_likelihood::Function
    params::Tuple
end

function pdf_likelihood(model::GeneralConjugateModel, y::Float64, θ::Tuple)
    model.pdf_likelihood(y, θ..., model.params...)
end
function pdf_likelihood(model::GeneralConjugateModel, y::Array{Float64,1}, θ::Tuple)
    model.pdf_likelihood(y, θ..., model.params...)
end
function sample_posterior(model::GeneralConjugateModel, y::Float64)
    model.sample_posterior(y, model.params...)
end
function sample_posterior(model::GeneralConjugateModel, y::Array{Float64,1})
    model.sample_posterior(y, model.params...)
end
function sample_posterior(model::GeneralConjugateModel, y::Array{Float64, 2})
    model.sample_posterior(y, model.params...)
end
function marginal_likelihood(model::GeneralConjugateModel, y::Float64)
    model.marginal_likelihood(y, model.params...)
end
