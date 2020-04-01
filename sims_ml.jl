######################################################################################################################
# Author: Alex Keil
# Program: sims_ml.R
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

@everywhere include("Gibbs.jl") # most functions not necessary here, but calls in some required packages

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


# full model
@everywhere function analyze(n=100)
  y,Xi,X2i,Zi,Xinti,Xint15i,y1,y15 = dgm(n);
  X =  hcat(ones(n), Xi, X2i, Zi, Zi .* Zi, Xi .* Zi, X2i .* Zi);
  Xint =  [hcat(ones(n), Xinti, Xinti, Zi, Zi .* Zi, Xinti .* Zi, Xinti .* Zi), 
           hcat(ones(n), Xint15i, Xint15i, Zi, Zi .* Zi, Xint15i .* Zi, Xint15i .* Zi)];
  res = fit(GeneralizedLinearModel,X,y, Normal())
  m1e, m0e = mean(predict(res, Xint[1])),mean(predict(res, Xint[2]))
  m1 = mean(y1)
  m0 = mean(y15)
  hcat(
    m1,m0, m1-m0,
    m1e, m0e, m1e-m0e
  )
end


# misspecified model
@everywhere function analyze2(n=100)
  y,Xi,X2i,Zi,Xinti,Xint15i,y1,y15 = dgm(n);
  X =  hcat(ones(n), Xi, X2i, Zi);
  Xint =  [hcat(ones(n), Xinti, Xinti, Zi), hcat(ones(n), Xint15i, Xint15i, Zi)];
  res = fit(GeneralizedLinearModel,X,y, Normal())
  m1e, m0e = mean(predict(res, Xint[1])),mean(predict(res, Xint[2]))
  m1 = mean(y1)
  m0 = mean(y15)
  hcat(
    m1,m0, m1-m0,
    m1e, m0e, m1e-m0e
  )
end


# trigger JIT compiler
rti = analyze(100)

Niter=2
sampsize=100
res = SharedArray{Float64}(Niter,size(rti)[2], pids=procs(myid()))
@inbounds @sync @distributed for i in 1:Niter
  res[i,:] = analyze(sampsize)
  #res[i,:] = analyze2(sampsize, chains=1)
  #res[i,:] = analyze3(sampsize, chains=1)
  println("testing")
end


Niter=2000
println(string(Niter)*" iterations")


sampsize=100
println("Sample size="*string(sampsize))
res = SharedArray{Float64}(Niter,size(rti)[2], pids=procs(myid()));
@inbounds @sync @distributed for i in 1:Niter
  res[i,:] = analyze(sampsize)
end
resfin = DataFrame(res);
rename!(resfin, [:m1t, :m0t, :mdt, :m1, :m0, :md]);
CSV.write("n100mle.csv", resfin)


sampsize=1000
println("Sample size="*string(sampsize))
res = SharedArray{Float64}(Niter,size(rti)[2], pids=procs(myid()));
@inbounds @sync @distributed for i in 1:Niter
  res[i,:] = analyze(sampsize)
end
resfin = DataFrame(res);
rename!(resfin, [:m1t, :m0t, :mdt, :m1, :m0, :md]);
CSV.write("n1000mle.csv", resfin)

sampsize=10000
println("Sample size="*string(sampsize))
res = SharedArray{Float64}(Niter,size(rti)[2], pids=procs(myid()));
@inbounds @sync @distributed for i in 1:Niter
  res[i,:] = analyze(sampsize)
end
resfin = DataFrame(res);
rename!(resfin, [:m1t, :m0t, :mdt, :m1, :m0, :md]);
CSV.write("n10000mle.csv", resfin)



sampsize=5000
println("Sample size="*string(sampsize))
res = SharedArray{Float64}(Niter,size(rti)[2], pids=procs(myid()));

@inbounds @sync @distributed for i in 1:Niter
  res[i,:] = analyze(sampsize)
end
resfin = DataFrame(res);
rename!(resfin, [:m1t, :m0t, :mdt, :m1, :m0, :md]);
CSV.write("n5000mle.csv", resfin)



println("Misspecified")
sampsize=10000
println("Sample size="*string(sampsize))
res = SharedArray{Float64}(Niter,size(rti)[2], pids=procs(myid()));
@inbounds @sync @distributed for i in 1:Niter
  res[i,:] = analyze2(sampsize)
end
resfin = DataFrame(res);
rename!(resfin, [:m1t, :m0t, :mdt, :m1, :m0, :md]);
CSV.write("n10000mle_misspec.csv", resfin)


sampsize=100
println("Sample size="*string(sampsize))
res = SharedArray{Float64}(Niter,size(rti)[2], pids=procs(myid()));
@inbounds @sync @distributed for i in 1:Niter
  res[i,:] = analyze2(sampsize)
end
resfin = DataFrame(res);
rename!(resfin, [:m1t, :m0t, :mdt, :m1, :m0, :md]);
CSV.write("n100mle_misspec.csv", resfin)


sampsize=1000
println("Sample size="*string(sampsize))
res = SharedArray{Float64}(Niter,size(rti)[2], pids=procs(myid()));
@inbounds @sync @distributed for i in 1:Niter
  res[i,:] = analyze2(sampsize)
end
resfin = DataFrame(res);
rename!(resfin, [:m1t, :m0t, :mdt, :m1, :m0, :md]);
CSV.write("n1000mle_misspec.csv", resfin)

sampsize=5000
println("Sample size="*string(sampsize))
res = SharedArray{Float64}(Niter,size(rti)[2], pids=procs(myid()));
@inbounds @sync @distributed for i in 1:Niter
  res[i,:] = analyze2(sampsize)
end
resfin = DataFrame(res);
rename!(resfin, [:m1t, :m0t, :mdt, :m1, :m0, :md]);
CSV.write("n5000mle_misspec.csv", resfin)

