

function benchmark(dataset::Array{Float64}, model::AbstractMixtureModel;α=1.0, iters=10000)
    @time dp_cluster(dataset, model, α, iters=iters)
    return
end
