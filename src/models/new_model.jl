#
#   Guidelines for how to create a model object:
#

# A struct to define the base type for your model (note, this can be empty if
# needed, its primary purpose is for dispatch)
struct MyModel <: UnivariateConjugateModel  #or MultivariateConjugateModel as appropriate

end

# Include whichever default constructors you want in addition to the base inner
# constructor provided by Julia
function MyModel() end

# Define the functions pdf_likelihood, sample_posterior and marginal_likelihood.
# Note that sample_posterior should accept either a single datapoint or a list
# of datapoints.
# In the univariate case this should be either a Float64 or an Array{Float64, 1}.
# In the multivariate case it should accept either an Array{Float64,1} or
# Array{Float64,2}.
# The first argument to each function should be an object of your model's type.
# This is used for dispatch, whether or not your functions actually need to access
# any of the model's hyperparameters.
function pdf_likelihood(model::MyModel, y::Float64, θ::Tuple) end
function sample_posterior(model::MyModel, Y::Array{Float64,1}) end
function sample_posterior(model::MyModel, y::Float64) end
function marginal_likelihood(model::MyModel, y::Float64) end

# to_string is an optional function you an define for your model to allow it to
# print the cluster parameters in a particular way. This is used only for the
# `summarize` function.
function to_string(model::MyModel, ϕ::Tuple)end
