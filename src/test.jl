include("DirichletMixtureModels.jl")

using Atom
using DirichletMixtureModels

import Distributions:
        UnivariateDistribution,
        MultivariateDistribution,
        Exponential,
        Normal,
        MvNormal,
        NormalStats,
        rand,
        suffstats

importall DirichletMixtureModels


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

srand(1010)

#Univariate Test Code
u_params = [(0.0,0.5), (0.5, 0.1)]
u_data = generateSamples(Normal, u_params, [100,100])

#   Multivariate Test Code
T1 = [2. 1.; 1. 2.]
T2 = [4. 0.5; 0.5 1.]
μ1 = [5.,0.]
μ2 = [0.,-5.]

m_params = [(μ1, T1), (μ2, T2)]
m_data = generateSamples(MvNormal, m_params, [100, 100], 2)

α=1.0

u_stats = suffstats(Normal, u_data)
u_model = UnivariateNormalModel(u_stats)
u_states = dp_cluster(u_data,u_model,1.0,iters=300)

m_stats = suffstats(MvNormal, m_data)
m_model = MultivariateNormalModel(m_stats)
m_states = dp_cluster(m_data,m_model,1.0,iters=300)


#=
#   Univariate Test Code

params = [(0.0, 1.0), (3.0, 1.0), (10.0, 1.0)]
data = generateSamples(Normal, params, [100, 100, 100])

α=1.0

ss = suffstats(Normal,data)
U = DMM.UnivariateNormalKnownSigma(ss, 1.0)
s = DMM.DPCluster(data,U,1.0)
=#
