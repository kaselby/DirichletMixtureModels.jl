#Redefining several methods from ConjugatePriors until it gets updated

@suppress begin
    function rand(nw::NormalWishart)
        Lam = rand(Wishart(nw.nu, nw.Tchol))
        Lsym = PDMat(Symmetric(inv(Lam) ./ nw.kappa))
        mu = rand(MvNormal(nw.mu, Lsym))
        return (mu, Lam)
    end

    function posterior_canon(prior::NormalWishart, ss::MvNormalStats)
        mu0 = prior.mu
        kappa0 = prior.kappa
        TC0 = prior.Tchol
        nu0 = prior.nu

        kappa = kappa0 + ss.tw
        nu = nu0 + ss.tw
        mu = (kappa0.*mu0 + ss.s) ./ kappa

        z = prior.zeromean ? ss.m : ss.m - mu0
        Lam = Symmetric(inv(inv(TC0) + ss.s2 + kappa0*ss.tw/kappa*(z*z')))

        return NormalWishart(mu, kappa, cholfact(Lam), nu)
    end
    pdf(nw::NormalWishart, x::Vector{T}, Lam::Matrix{S}) where T<:Real where S<:Real =
    exp(logpdf(nw, x, Lam))
    function logpdf(nw::NormalWishart, x::Vector{T}, Lam::Matrix{T}) where T<:Real
        p = length(x)
        nu = nw.nu
        kappa = nw.kappa
        mu = nw.mu
        Tchol = nw.Tchol
        hnu = 0.5 * nu
        hp = 0.5 * p

        # Normalization
        logp = hp*(log(kappa) - Float64(log(2*π)))
        logp -= hnu * logdet(Tchol)
        logp -= hnu * p * log(2.)
        logp -= logmvgamma(p, hnu)

        # Wishart (MvNormal contributes 0.5 as well)
        logp += (hnu - hp) * logdet(Lam)
        logp -= 0.5 * trace(Tchol \ Lam)

        # Normal
        z = nw.zeromean ? x : x - mu
        logp -= 0.5 * kappa * dot(z, Lam * z)

        return logp
    end
end
