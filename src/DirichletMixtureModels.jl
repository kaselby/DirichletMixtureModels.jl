module DirichletMixtureModels

    using Distributions
    using ConjugatePriors
    using PDMats

    import PDMats: PDMat

    import Distributions:
        Distribution,
        UnivariateDistribution,
        MultivariateDistribution,
        NormalCanon,
        Normal,
        NormalKnownSigma,
        Gamma,
        Exponential,
        MvNormal,
        MvNormalCanon,
        GenericMvTDist,
        MvNormalStats,
        NormalStats,
        logmvgamma,
        suffstats,
        pdf

    import ConjugatePriors:
        NormalGamma,
        NormalWishart,
        rand,
        pdf,
        logpdf,
        posterior_canon

    export
        AbstractMixtureModel,
        ConjugateModel,
        UnivariateConjugateModel,
        MultivariateConjugateModel,
        UnivariateExponentialModel,
        UnivariateNormalKnownSigma,
        UnivariateNormalModel,
        MultivariateNormalModel,
        DMMState,
        OutputState,

        dp_cluster,
        summarize,
        export_states,
        pdf_likelihood,
        sample_posterior,
        marginal_likelihood,
        standard_form,
        parameter_names,
        benchmark


    include("./package_overrides.jl")
    include("./model.jl")
    include("./models/nonconjugate_model.jl")
    include("./models/general_model.jl")
    include("./models/uv_normal.jl")
    include("./models/mv_normal.jl")
    include("./models/uv_exp.jl")
    include("./DMMState.jl")
    include("./DPCluster.jl")
    include("./benchmark.jl")

end # module
