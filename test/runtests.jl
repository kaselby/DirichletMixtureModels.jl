using DirichletMixtureModels
using Distributions
using Base.Test

import DirichletMixtureModels:
    UnivariateNormalModel,
    MultivariateNormalModel,
    marginal_likelihood,
    pdf_joint,
    pdf_posterior,
    sample_prior
import Distributions:
    Normal,
    MvNormal,
    suffstats

function test_model(model, y, θ)
    marginal_likelihood(model, y) - (pdf_joint(model, y, θ) / pdf_posterior(model, y, θ))
end

TOL = 1e-8

N_θ = 10
N_y = 10

data_1d = randn(N_y)
data_2d = randn(2,N_y)

ss_1d = suffstats(Normal, data_1d)
ss_2d = suffstats(MvNormal, data_2d)

uvn_model = UnivariateNormalModel(0., 1., 1., 1.)
mvn_model = MultivariateNormalModel([0.,0.], 1., [1. 0.; 0. 1.], 2.)

for i in 1:N_θ
    θ_1d = sample_prior(uvn_model)
    θ_2d = sample_prior(mvn_model)
    for j in 1:N_y
        y_1d = randn()
        y_2d = randn(2)
        try
            @test abs(test_model(uvn_model, y_1d, θ_1d)) < TOL
            @test abs(test_model(mvn_model, y_2d, θ_2d)) < TOL
        catch
            println("$y_1d, $θ_1d, $(test_model(uvn_model, y_1d, θ_1d))")
            println("$y_2d, $θ_2d, $(test_model(mvn_model, y_2d, θ_2d))")
            raise()
        end
    end
end
