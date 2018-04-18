abstract type AbstractMixtureModel end
abstract type ConjugateModel <: AbstractMixtureModel end
abstract type UnivariateConjugateModel <: ConjugateModel end
abstract type MultivariateConjugateModel <: ConjugateModel end

#   Fallback
function to_string(model::AbstractMixtureModel, ϕ::Tuple)
    return string(ϕ)
end
