
using DirichletMixtureModels
using DataFrames
using CSV
importall DirichletMixtureModels

DATAPOINTS = [50,100,200,400]
DIMS = [1,2,3,4]

data = randn(DATAPOINTS[end])

times = zeros(length(DATAPOINTS))

# Univariate case

for j in 1:length(DATAPOINTS)
    N=DATAPOINTS[j]
    data_N = data[1:N]
    uv_model = UnivariateNormalModel(data_N)

    #throwaway
    dp_benchmark(zeros(1), uv_model, iters=100)

    t=dp_benchmark(data_N, uv_model, iters=100)
    println("N=$N, d=1, t=$t")
    times[j] = t
end

# Multivariate case
for i in 2:length(DIMS)
    d=DIMS[i]
    for j in 1:length(DATAPOINTS)
        N=DATAPOINTS[j]
        data_dN = data[1:d,1:N]
        mv_model = MultivariateNormalModel(data_dN)

        #throwaway
        dp_benchmark(zeros(d,1), mv_model)

        t=dp_benchmark(data_dN, mv_model)
        println("N=$N, d=$d, t=$t")
        times[i,j] = t
    end
end

head=DataFrame(DATAPOINTS)
CSV.write("benchmarks3.csv",head,header=false)
timesdata=convert(DataFrame, times)
CSV.write("benchmarks3.csv",timesdata,append=true,header=false)
