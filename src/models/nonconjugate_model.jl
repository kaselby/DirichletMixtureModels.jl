struct NonConjugateModel <: AbstractMixtureModel
    pdf_likelihood::Function
    sample_prior::Function
    sample_posterior::Nullable{Function}
    params::Tuple
end

function GeneralNonConjugateModel(pdf_l::Function, s_prior::Function, params::Tuple)
    GeneralNonConjugateModel(pdf_l, s_prior, Nullable{Function}(), params)
end
function GeneralNonConjugateModel(pdf_l::Function, s_prior::Function, s_post::Function, params::Tuple)
    GeneralNonConjugateModel(pdf_l, s_prior, Nullable{Function}(s_post), params)
end

function pdf_likelihood(model::NonConjugateModel, y::Array{Float64,1}, θ::Tuple)
    model.pdf_likelihood(y, θ..., model.params...)
end
function sample_prior(model::NonConjugateModel)
    model.sample_prior(model.params...)
end
function sample_posterior(model::NonConjugateModel, y::Union{Float64, Array{Float64,1}, Array{Float64, 2}}, m::Int64)
    if isnull(model.sample_posterior)
        mc_sample_posterior(model, y, m)
    else
        get(model.sample_posterior)(y, model.params...)
    end
end

function mc_sample_posterior(model::NonConjugateModel, y::Union{Float64, Array{Float64,1}, Array{Float64,2}}, m::Int64)
    aux=Array{Tuple}(m)
    for i=1:m
        aux[i] = sample_prior(model)
    end
    q_aux=[pdf_likelihood(model, y, θ) for θ in aux]
    b=sum(q_aux)
    q_aux/=b

    rd=rand()
    p=0
    for i in 1:m
        p += q_aux[i]
        if rd < p
            return aux[i]
        end
    end
    error("This statement should never be reached.")
end
