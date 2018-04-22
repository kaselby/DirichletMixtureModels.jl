
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

function generateSamples(::Type{T}, thetas::AbstractVector, numSamples::Array{Int64,1}; shuffled=true) where T <: UnivariateDistribution
    @assert length(thetas) == length(numSamples)
    M=length(thetas)
    N=sum(numSamples)
    data=zeros(Float64, N)
    n=0
    for i in 1:M
        n_i=numSamples[i]
        dist_i=T(thetas[i]...)
        for j in 1:n_i
            data[n+j]=rand(dist_i)
        end
        n+=n_i
    end
    if shuffled
        shuffle!(data)
    end
    return data
end
function generateSamples(::Type{T}, thetas::AbstractVector, numSamples::Array{Int64,1}, d::Int64; shuffled=true) where T <: MultivariateDistribution
    @assert length(thetas) == length(numSamples)
    M=length(thetas)
    N=sum(numSamples)
    data=zeros(Float64, d, N)
    n=0
    for i in 1:M
        n_i=numSamples[i]
        dist_i=T(thetas[i]...)
        for j in 1:n_i
            data[:,n+j]=rand(dist_i)
        end
        n+=n_i
    end
    if shuffled
        data=data[:,shuffle(1:end)]
    end
    return data
end
