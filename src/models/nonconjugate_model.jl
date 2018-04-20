"""
    NonConjugateModel(pdf_likelihood, sample_prior, params)
This is the base type for a general model with non-conjugate likelihood and prior.
The arguments should be functions of the following form
```julia
pdf_likelihood(y, θ_1, ..., θ_n, η_1, ..., η_m)
sample_prior(η_1, ..., η_m)
```
Where θ_1,...,θ_n are the parameters for the likelihood and `params=(η_1,...,η_m)`.
pdf_likelihood should return a float when supplied with one data point and an
array of floats when supplied with a set of data points. sample_prior should return
a sample from the prior of the form (θ_1,...,θ_n).
"""
struct NonConjugateModel <: AbstractMixtureModel
    pdf_likelihood::Function
    sample_prior::Function
    params::Tuple
end


function pdf_likelihood(model::NonConjugateModel, y::Float64, θ::Tuple)
    model.pdf_likelihood(y, θ..., model.params...)
end
function pdf_likelihood(model::NonConjugateModel, y::Array{Float64,1}, θ::Tuple)
    prod(model.pdf_likelihood(y, θ..., model.params...))
end
function sample_prior(model::NonConjugateModel)
    model.sample_prior(model.params...)
end
