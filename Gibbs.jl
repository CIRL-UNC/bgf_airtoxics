######################################################################################################################
# Author: Alex Keil
# Program: Gibbs.jl
# Language: Julia (tested on v1.3.0)
# Date: Tuesday, March 31, 2020 at 3:12:51 PM
# Project: Bayesian G-Computation to Estimate Impacts of Interventions on Exposure Mixtures: 
#    Demonstration with Metals from Coal-fired Power Plants and Birthweight
# Tasks:
# Data in: 
# Data out: 
# Description:
# Keywords: mixtures, bayes, g-computation, causal inference, coal, environment
# Released under the GNU General Public License: http://www.gnu.org/copyleft/gpl.html
######################################################################################################################

using Distributions, Random, DataFrames, GLM, StatsBase, LinearAlgebra, CSV, Distributed


# wrapping intercept into block sampler
function gibbsint(y, X, Xint, pl, iter, burnin, rng; thin=1, chain=1,
               _mu_eta0 = 0., _mu_eta1 = 10.,       # prior mean, sd of intercept
               _mu0 = 0., _mu1 = 1.,                # prior mean, sd of mu_l (beta means)
               _sigma0 = 0.0001, _sigma1 = 0.0001,  # prior a,b parameters for sigma (model error)
               _tau0 = 0.0001, _tau1=0.0001,          # prior a, b parameters for tau (beta standard deviations)
               mcbirthweight = 3190.891746301147577469, # rescaling coefficients
               scbirthweight = 631.3238622088545071165 # rescaling coefficients
               )
  # block sampler
  (N,p) = size(X)
  X = hcat(ones(N), X)
  Xint = [hcat(ones(N), Xint[1]), hcat(ones(N), Xint[2])]
  p = p+1
  j = size(pl)[1]
  if sum(pl) != p
    throw("sum(pl) should equal number of columns of X (without intercept)")
  end
  # constants/hyperpriors
  # initial values
  _sigma = rand()*2
  _beta = rand(p)
  _tau = rand(j)
  _mu = rand(j)
  _beta_store = zeros(iter, p)
  _mu_store = zeros(iter, j)
  _tau_store = zeros(iter, j)
  _sigma_store = zeros(iter)
  m_store = zeros(iter, 3)
  Xt = transpose(X)
  xtx = Xt * X
  munc, muint = ones(N), ones(N)
  @inbounds for i in 1:iter
    ####################
    # update sigma
    ####################
    se = (y .- X * _beta).^2. #  permutedims(y .- X * _beta) * (y .- X * _beta)
    a = _sigma0 + N/2.
    b = _sigma1 + sum(se)/2.
    _sigma = sqrt(rand(rng, InverseGamma(a, b)))
    ####################
    # update tau
    ####################
    for l in range(1,stop=j)
      stidx = l > 1 ? sum(pl[1:(l-1)])+1 : 1
      endidx = stidx + pl[l] -1
      bl = _beta[stidx:endidx]
      bse = (bl .- _mu[l]) .^2
      a_tau = _tau0 + pl[l]/2.
      b_tau = _tau1 + sum(bse)/2.
      _tau[l] = sqrt(rand(rng, InverseGamma(a_tau, b_tau)))
    end
    ####################
    # update mu
    ####################
    for l in range(1,stop=j)
      stidx = l > 1 ? sum(pl[1:(l-1)])+1 : 1
      endidx = stidx + pl[l] -1
      bl = _beta[stidx:endidx]
      V = inv(pl[l] * _tau[l].^(-2) +  _mu1 .^(-2))
      M = V * (transpose(bl)*ones(pl[l]) * _tau[l].^(-2) .+ _mu0 * _mu1 .^(-2))
      _mu[l] = rand(rng, Normal(M, sqrt(V)))
    end
    ####################
    # update beta
    ####################
    # expand Lam, mu to pXp matrix
    Lam = Diagonal(vcat([ ones(pl[l])* _tau[l] .^2 for l in 1:j]...))
    _muvec = vcat([ ones(pl[l])*_mu[l] for l in 1:j]...)
    iLam = inv(Lam)
    V = Symmetric(inv(xtx .* _sigma^(-2)  + iLam))
    M = V * (Xt * y ./_sigma^2 + iLam * _muvec)
    _beta = rand(rng, MvNormal(M, V))
    ####################
    # update mean difference
    ####################
    #@inbounds for k in 1:N
    #  munc[k] =  (Xint[1][k:k,:] * _beta)[1]
    #  muint[k] = (Xint[2][k:k,:] * _beta)[1]
    #end
    munc =  Xint[1] * _beta
    muint = Xint[2] * _beta
    cm1 = mcbirthweight + scbirthweight*mean(munc)
    cm0 = mcbirthweight + scbirthweight*mean(muint)
    md = cm1-cm0
    ####################
    # store sweep values
    ####################    
    _sigma_store[i] = _sigma
    _beta_store[i,:] = _beta
    _mu_store[i,:] = _mu
    _tau_store[i,:] = _tau
    m_store[i,:] = vcat(cm1, cm0, md)
  end
  df = convert(DataFrame, hcat([chain for i in 1:iter], [i for i in 1:iter], m_store, _beta_store, _sigma_store,_mu_store,_tau_store))
  rename!(df, vcat(
       :chain, :iter,
       :m1, :m0, :md,
     [Symbol("b" * "[$i]") for i in 0:(p-1)],
     :sigma,
     [Symbol("mub" * "[$i]") for i in 1:(j)],
     [Symbol("taub" * "[$i]") for i in 1:(j)]
     ))
  df[range(burnin+1, iter, step=thin),:]
end

gibbsint(y,X,Xint,pl, iter, burnin;thin=1,chain=1,_mu_eta0 = 0., _mu_eta1 = 10.,_mu0 = 0., _mu1 = 1.,_sigma0 = 0.0001, _sigma1 = 0.0001, _tau0 = 0.0001, _tau1=0.0001, mcbirthweight = 3190.891746301147577469, scbirthweight = 631.3238622088545071165) = gibbsint(y,X,Xint,pl, iter, burnin, MersenneTwister(convert(Int, rand([i for i in 1:1e6])));thin=thin,chain=chain, _mu_eta0 = _mu_eta0, _mu_eta1 = _mu_eta1, _mu0 = _mu0, _mu1 = _mu1, _sigma0 = _sigma0, _sigma1 = _sigma1, _tau0=  _tau0, _tau1=_tau1, mcbirthweight = mcbirthweight, scbirthweight = scbirthweight)
gibbsint(y,X,Xint,pl, iter;thin=1,chain=1,_mu_eta0 = 0., _mu_eta1 = 10.,_mu0 = 0., _mu1 = 1.,_sigma0 = 0.0001, _sigma1 = 0.0001, _tau0 = 0.0001, _tau1=0.0001, mcbirthweight = 3190.891746301147577469, scbirthweight = 631.3238622088545071165) = gibbsint(y,X,Xint,pl, iter, 0;thin=thin,chain=chain, _mu_eta0 = _mu_eta0, _mu_eta1 = _mu_eta1, _mu0 = _mu0, _mu1 = _mu1, _sigma0 = _sigma0, _sigma1 = _sigma1, _tau0=  _tau0, _tau1=_tau1, mcbirthweight = mcbirthweight, scbirthweight = scbirthweight)



function summarymcmc(results::DataFrame)
 sets, means, medians, pl, pu, stds, ac1, ac5, lens = Array[], Array[], Array[], Array[], Array[], Array[], Array[], Array[], Array[]
 nm = names(results)
 for i in 1:size(results, 2)
   col = results[:,i]
   means = vcat(means, mean(col))
   medians = vcat(medians, median(col))
   pl = vcat(pl, quantile(col, 0.025)[1])
   pu = vcat(pu, quantile(col,  0.975)[1])
   stds = vcat(stds, std(col))
   ac = autocor(col, [1,5])
   ac1 = vcat(ac1, ac[1])
   ac5 = vcat(ac5, ac[2])
   lens = vcat(lens, length(col))
 end
 res = convert(DataFrame, hcat(nm, means, stds, medians, pl, pu, ac1, ac5, lens))
 rename!(res, [:nm, :mean, :std, :median, :lower2_5, :upper97_5, :autocor_1, :autocor_5, :length])
 return res
end
