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
