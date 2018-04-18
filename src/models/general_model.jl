#
#   For use with R package
#


struct GeneralUnivariateConjugateModel <: UnivariateConjugateModel
    marginal_likelihood::Function
    pdf_likelihood::Function
    sample_posterior::Function
    marg_params::Tuple
    pdf_params::Tuple
    post_params::Tuple
end
struct GeneralMultivariateConjugateModel <: MultivariateConjugateModel
    marginal_likelihood::Function
    pdf_likelihood::Function
    sample_posterior::Function
    marg_params::Tuple
    pdf_params::Tuple
    post_params::Tuple
end


function GeneralUnivariateConjugateModel(ML::Function, SP::Function, PL::Function, params::Tuple)
    GeneralUnivariateConjugateModel(ML,PL,SP,params,params,params)
end
function GeneralMultivariateConjugateModel(ML::Function, SP::Function, PL::Function, params::Tuple)
    GeneralMultivariateConjugateModel(ML,PL,SP,params,params,params)
end


function pdf_likelihood(model::GeneralUnivariateConjugateModel, y::Float64, θ::Tuple)
    model.pdf_likelihood(y,θ...,pdf_params...)
end
function sample_posterior(model::GeneralUnivariateConjugateModel, y::Union{Float64,Array{Float64,1}})
    model.sample_posterior(y,post_params...)
end
function marginal_likelihood(model::GeneralUnivariateConjugateModel, y::Float64)
    model.marginal_likelihood(y,marg_params...)
end

function pdf_likelihood(model::GeneralMultivariateConjugateModel, y::Array{Float64,1}, θ::Tuple)
    model.pdf_likelihood(y,θ...,pdf_params...)
end
function sample_posterior(model::GeneralMultivariateConjugateModel, y::Union{Array{Float64,1},Array{Float64,2}})
    model.sample_posterior(y,post_params...)
end
function marginal_likelihood(model::GeneralMultivariateConjugateModel, y::Array{Float64,1})
    model.marginal_likelihood(y,marg_params...)
end
