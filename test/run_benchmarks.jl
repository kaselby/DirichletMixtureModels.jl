
using DirichletMixtureModels
importall DirichletMixtureModels

DATAPOINTS = [10,30,100,300]
DIMS = [1,2,3]


times = zeros(length(DIMS),length(DATAPOINTS))

# Univariate case
uv_model = UnivariateNormalModel()
for j in 1:length(DATAPOINTS)
    N=DATAPOINTS[j]
    data = randn(N)

    #throwaway
    benchmark(zeros(1), uv_model)

    t=benchmark(data, uv_model)
    println("1 $j $t")
    times[1,j] = t
end

# Multivariate case
for i in 2:length(DIMS)
    d=DIMS[i]
    mv_model = MultivariateNormalModel(d)
    for j in 1:length(DATAPOINTS)
        N=DATAPOINTS[j]
        data = randn(d,N)

        #throwaway
        t=benchmark(zeros(d,1), mv_model)

        t = benchmark(data, mv_model)
        println("$i $j $t")
        times[i,j] = t
    end
end
