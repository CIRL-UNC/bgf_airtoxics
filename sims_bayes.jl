######################################################################################################################
# Author: Alex Keil
# Program: sims_bayes.R
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


using Distributed

cd("/workingdirectory")

@everywhere include("Gibbs.jl")
@everywhere using Distributions, Random, SharedArrays


# data generating function
@everywhere function dgm(n)
  alpha = [2.0, 1.0, 0.0]
  Z = permutedims(rand(MvNormal([0.,0.,0.], 1.), n))
  X = 15. .+ Z * alpha + rand(Normal(0.0, 1.0), n)
  X2 = 15. .+ Z * alpha + rand(Normal(0.0, 1.0), n)
  Xint1 = X*0. .+ 1.0
  Xint15 = X*0. .+ 15.0
  beta = vcat(
  [1.0],            #x (x2 has no effect)
  [1.0, 0.5, 1.5], #z
  [-0.1, -0.15, -0.2], #z*z
  [-0.3, -0.25, -0.2]   #x*z
  )
  y = hcat(X, Z, Z .* Z, X .* Z) * beta + rand(Normal(0.0, 3.0), n)
  y1 = hcat(Xint1, Z, Z .* Z, Xint1 .* Z) * beta# + rand(Normal(0.0, 1.0), n)
  y15 = hcat(Xint15, Z, Z .* Z, Xint15 .* Z) * beta# + rand(Normal(0.0, 1.0), n)
  y,X,X2,Z,Xint1, Xint15,y1, y15
end


@everywhere function analyze(n=100;chains=2, iters=12000, burn=2000, thin=1)
  y,Xi,X2i,Zi,Xinti,Xint15i,y1,y15 = dgm(n);
  X =  hcat(Xi, X2i, Zi, Zi .* Zi, Xi .* Zi, X2i .* Zi);
  Xint =  [hcat(Xinti, Xinti, Zi, Zi .* Zi, Xinti .* Zi, Xinti .* Zi), 
           hcat(Xint15i, Xint15i, Zi, Zi .* Zi, Xint15i .* Zi, Xint15i .* Zi)];
  res = [gibbsint(y,X,Xint,[1,2,3,9], iters, burn, thin=thin, mcbirthweight = 0., scbirthweight = 1., chain=g, _mu0 = 0., _mu1 = 10.) for g in 1:chains];
  resg = vcat(res...);
  #CSV.write("sampint.csv", resg)
  op = summarymcmc(resg)
  m1 = mean(y1)
  m0 = mean(y15)
  hcat(
    m1,m0, m1-m0,
    op[!, :mean][op.nm .== :m1],
    op[!, :mean][op.nm .== :m0],
    op[!, :mean][op.nm .== :md],
    op[!, :std][op.nm .== :m1],
    op[!, :std][op.nm .== :m0],
    op[!, :std][op.nm .== :md],
    op[!, :lower2_5][op.nm .== :m1],
    op[!, :lower2_5][op.nm .== :m0],
    op[!, :lower2_5][op.nm .== :md],
    op[!, :upper97_5][op.nm .== :m1],
    op[!, :upper97_5][op.nm .== :m0],
    op[!, :upper97_5][op.nm .== :md]
  )
end


# trigger JIT compiler
rti = analyze(10, chains=1)

Niter=2
sampsize=100
res = SharedArray{Float64}(Niter,size(rti)[2], pids=procs(myid()))
@inbounds @sync @distributed for i in 1:Niter
  res[i,:] = analyze(sampsize, chains=1)
  println("testing")
end


Niter=1000
println(string(Niter)*" iterations")

sampsize=100
println("Sample size="*string(sampsize))
res = SharedArray{Float64}(Niter,size(rti)[2], pids=procs(myid()))
@inbounds @sync @distributed for i in 1:Niter
  res[i,:] = analyze(sampsize, iters=40000, burn=10000, chains=5, thin=5)
end
resfin = DataFrame(res);
rename!(resfin, [:m1t, :m0t, :mdt, :m1, :m0, :md, :s1, :s0, :sd, :m1l, :m0l, :mdl, :m1u, :m0u, :mdu]);
CSV.write("n100.csv", resfin)


sampsize=1000
println("Sample size="*string(sampsize))
res = SharedArray{Float64}(Niter,size(rti)[2], pids=procs(myid()))
@inbounds @sync @distributed for i in 1:Niter
  res[i,:] = analyze(sampsize, iters=40000, burn=10000, chains=5, thin=5)
end
resfin = DataFrame(res);
rename!(resfin, [:m1t, :m0t, :mdt, :m1, :m0, :md, :s1, :s0, :sd, :m1l, :m0l, :mdl, :m1u, :m0u, :mdu]);
CSV.write("n1000.csv", resfin)


sampsize=10000
println("Sample size="*string(sampsize))
res = SharedArray{Float64}(Niter,size(rti)[2], pids=procs(myid()))
@inbounds @sync @distributed for i in 1:Niter
  res[i,:] = analyze(sampsize, iters=40000, burn=10000, chains=5, thin=5)
end
resfin = DataFrame(res);
rename!(resfin, [:m1t, :m0t, :mdt, :m1, :m0, :md, :s1, :s0, :sd, :m1l, :m0l, :mdl, :m1u, :m0u, :mdu]);
CSV.write("n10000.csv", resfin)



