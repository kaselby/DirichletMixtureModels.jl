
struct GeneralUnivariateConjugateModel <: UnivariateConjugateModel
    marginal_likelihood::Function
    pdf_likelihood::Function
    sample_posterior::Function
    params::Tuple
end
struct GeneralMultivariateConjugateModel <: MultivariateConjugateModel
    marginal_likelihood::Function
    pdf_likelihood::Function
    sample_posterior::Function
    params::Tuple
end

function pdf_likelihood(model::GeneralUnivariateConjugateModel, y::Float64, θ::Tuple)
    model.pdf_likelihood(y,θ...,params...)
end
function sample_posterior(model::GeneralUnivariateConjugateModel, y::Union{Float64,Array{Float64,1}})
    model.sample_posterior(y,params...)
end
function marginal_likelihood(model::GeneralUnivariateConjugateModel, y::Float64)
    model.marginal_likelihood(y,params...)
end

function pdf_likelihood(model::GeneralMultivariateConjugateModel, y::Array{Float64,1}, θ::Tuple)
    model.pdf_likelihood(y,θ...,params...)
end
function sample_posterior(model::GeneralMultivariateConjugateModel, y::Union{Array{Float64,1},Array{Float64,2}})
    model.sample_posterior(y,params...)
end
function marginal_likelihood(model::GeneralMultivariateConjugateModel, y::Array{Float64,1})
    model.marginal_likelihood(y,params...)
end
