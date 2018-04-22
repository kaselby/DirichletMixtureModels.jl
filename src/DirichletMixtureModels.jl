module DirichletMixtureModels

    using Suppressor
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
        GeneralConjugateModel,
        NonConjugateModel,
        UnivariateExponentialModel,
        UnivariateNormalKnownSigma,
        UnivariateNormalModel,
        MultivariateNormalModel,
        DMMState,

        dp_cluster,
        summarize,
        export_r,
        export_r_all,
        pdf_likelihood,
        sample_posterior,
        sample_prior,
        marginal_likelihood,
        standard_form,
        parameter_names,
        dp_benchmark


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
    include("./test_functions.jl")
end # module
