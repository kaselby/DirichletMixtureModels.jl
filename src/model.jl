abstract type AbstractMixtureModel end
abstract type ConjugateModel <: AbstractMixtureModel end
abstract type UnivariateConjugateModel <: ConjugateModel end
abstract type MultivariateConjugateModel <: ConjugateModel end
