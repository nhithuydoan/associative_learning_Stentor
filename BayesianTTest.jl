# Bayesian t-test (Rouder et al., 2009)
#
# Provides Bayes factor in favor of the null hypothesis (B01)
# for one-sample and two-sample independent t-tests with a
# Cauchy(r) prior on standardized effect size.
#
# Requirements: Julia 1.9+; standard library only plus QuadGK
#   import Pkg; Pkg.add(“QuadGK”)
module BayesianTTest
using Statistics
using QuadGK
export bayesian_ttest
"""
    bayesian_ttest(x; y=nothing, r=1.0)
Compute the Bayes factor *B₀₁* (favoring the null over the alternative)
for a t-test following Rouder et al. (2009), using a Cauchy prior on the
standardized effect size with scale `r` (default `r = 1.0`).
Arguments
---------
- `x::AbstractVector{<:Real}`: sample data (one-sample) or group 1 (two-sample)
- `y::Union{Nothing, AbstractVector{<:Real}}`: if provided, group 2 (two-sample); otherwise one-sample test
- `r::Real`: Cauchy prior scale for effect size (default 1.0). Common choices include 0.707 (“medium”)
Returns
-------
- `B01::Float64`: Bayes factor in favor of the null hypothesis (H0) versus the alternative (H1).
Notes
-----
- Two-sample case assumes independent samples with equal variances (the classic Student t-test).
- For one-sample, we use `n = length(x)`, degrees of freedom `ν = n-1`, and `N = n`.
- For two-sample, `ν = n₁ + n₂ - 2` and `N = (n₁*n₂)/(n₁+n₂)`.
References
----------
Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D., & Iverson, G. (2009).
Bayesian t tests for accepting and rejecting the null hypothesis.
*Psychonomic Bulletin & Review, 16*(2), 225–237.
"""
function bayesian_ttest(x::AbstractVector{<:Real}; y::Union{Nothing,AbstractVector{<:Real}}=nothing, r::Real=1.0)
    @assert length(x) ≥ 2 "x must have at least 2 observations"
    @assert r > 0 "r (Cauchy scale) must be positive"
    # Compute t statistic, degrees of freedom ν, and effective sample size N
    # t::Float64
    # ν::Float64
    # N::Float64
    if y === nothing
        n = length(x)
        μ̂ = mean(x)
        s = std(x; corrected=true)  # sample std with Bessel’s correction
        t = μ̂ / (s / sqrt(n))
        ν = n - 1
        N = n
    else
        @assert length(y) ≥ 2 "y must have at least 2 observations"
        n1 = length(x)
        n2 = length(y)
        μ1 = mean(x)
        μ2 = mean(y)
        s1 = var(x; corrected=true)
        s2 = var(y; corrected=true)
        ν = n1 + n2 - 2
        sp2 = ((n1 - 1) * s1 + (n2 - 1) * s2) / ν
        se = sqrt(sp2 * (1/n1 + 1/n2))
        t = (μ1 - μ2) / se
        N = (n1 * n2) / (n1 + n2)
    end
    # Numerator term under H0
    numerator = (1 + (t^2) / ν) ^ (-(ν + 1) / 2)
    # Denominator integrates over g where effect size δ ~ Cauchy(0, r)
    # using the Jeffreys-Zellner-Siow (JZS) formulation: g ~ InvGamma(1/2, 1/2)
    integrand(g) = begin
        denom = 1 + N * g * r^2
        term1 = denom^(-0.5)
        term2 = (1 + (t^2) / (denom * ν)) ^ (-(ν + 1) / 2)
        term3 = (2π)^(-0.5) * g^(-1.5) * exp(-1 / (2g))
        term1 * term2 * term3
    end
    # Integrate from 0 to ∞. QuadGK supports infinite limits.
    denominator, _ = quadgk(integrand, 0.0, Inf; rtol=1e-8, atol=1e-12, maxevals=10^7)
    B01 = numerator / denominator
    return B01
end
end 
# module
# -----------------------
# Example usage (uncomment to run):
# using .BayesianTTest
# x = randn(30) .+ 0.2           # one-sample example
# println(bayesian_ttest(x; r=0.707))
# y = randn(28) .+ 0.0
# println(bayesian_ttest(x; y=y, r=0.707))