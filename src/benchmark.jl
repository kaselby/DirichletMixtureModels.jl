

function benchmark(dataset::Array{Float64}, model::AbstractMixtureModel;α=1.0, iters=10000)
    @elapsed dp_cluster(dataset, model, α, iters=iters)
end
