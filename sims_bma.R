######################################################################################################################
# Author: Alex Keil
# Program: sims_bma.R
# Language: R (tested on v 3.6.1)
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

jpath = "~/bin/jags"
wd = "/workingdirectory"


setwd(wd)

library(R2jags)
library(future)
library(future.apply)
library(qgcomp)

# data generating function
dgm <- function(n){
  alpha = c(2.0, 1.0, 0.0)
  Z = qgcomp:::.rmvnorm(n, c(0,0,0), diag(rep(1, 3)))
  X = 15. + Z %*% alpha + rnorm(n)
  X2 = 15. + Z %*% alpha + rnorm(n)
  Xint1 = X*0. + 1.0
  Xint15 = X*0. + 15.0
  beta = c(
    1.0,            #x (x2 has no effect)
    1.0, 0.5, 1.5, #z
    -0.1, -0.15, -0.2, #z*z
    -0.3, -0.25, -0.2   #x*z
  )
  y = cbind(X, Z, Z * Z, cbind(X, X, X) * Z) %*% beta + rnorm(n, 0, 3)
  y1 = cbind(Xint1, Z, Z * Z, cbind(Xint1, Xint1, Xint1) * Z) %*% beta# + rand(Normal(0.0, 1.0), n)
  y15 = cbind(Xint15, Z, Z * Z, cbind(Xint15, Xint15, Xint15) * Z) %*% beta# + rand(Normal(0.0, 1.0), n)
  res = list(
    obs = data.frame(y=y,x=X,x2=X2,z=Z),
    int1 = data.frame(y=y1,x=Xint1,x2=Xint1,z=Z),
    int0 = data.frame(y=y15,x=Xint15,x2=Xint15,z=Z)
  )
  names(res[[1]]) <- names(res[[2]]) <- names(res[[3]]) <- c(
    "y", "x", "x2", paste0("z", 1:3)
  )
  res
}

jagsmod = "
  data {
    # can do tranformations here in future
    dummy = 1
  }
  model {
    for(i in 1:N){
       mu[i] <- (b0 + b[1]*x[i] + b[2]*x2[i] + 
            b[3]*z1[i]+ b[4]*z2[i]+ b[5]*z3[i] +
            b[6]*z1[i]*z1[i]+ b[7]*z2[i]*z2[i]+ b[8]*z3[i]*z3[i] + 
            b[9]*x[i]*z1[i]+ b[10]*x[i]*z2[i]+ b[11]*x[i]*z3[i] + 
            b[12]*x2[i]*z1[i]+ b[13]*x2[i]*z2[i]+ b[14]*x2[i]*z3[i] + 
            0
            )
       muint1[i] <- (b0 + b[1]*xint1[i] + b[2]*xint1[i] + 
            b[3]*z1[i]+ b[4]*z2[i]+ b[5]*z3[i] +
            b[6]*z1[i]*z1[i]+ b[7]*z2[i]*z2[i]+ b[8]*z3[i]*z3[i] + 
            b[9]*xint1[i]*z1[i]+ b[10]*xint1[i]*z2[i]+ b[11]*xint1[i]*z3[i] + 
            b[12]*xint1[i]*z1[i]+ b[13]*xint1[i]*z2[i]+ b[14]*xint1[i]*z3[i] + 
            0
            )
       muint0[i] <- (b0 + b[1]*xint0[i] + b[2]*xint0[i] + 
            b[3]*z1[i]+ b[4]*z2[i]+ b[5]*z3[i] +
            b[6]*z1[i]*z1[i]+ b[7]*z2[i]*z2[i]+ b[8]*z3[i]*z3[i] + 
            b[9]*xint0[i]*z1[i]+ b[10]*xint0[i]*z2[i]+ b[11]*xint0[i]*z3[i] + 
            b[12]*xint0[i]*z1[i]+ b[13]*xint0[i]*z2[i]+ b[14]*xint0[i]*z3[i] + 
            0
            )
       mdi[i] <- muint1[i] - muint0[i]
       y[i] ~ dnorm(mu[i], sigma)
    }
    # effect measures
    md <- mean(mdi)
    m1 <- mean(muint1)
    m0 <- mean(muint0)

    # prior model variance
    sigma ~ dt(0, 1, 1) T(0,) # half cauchy prior 
    # prior probability of exclusion
    pi[1] ~ dbeta(1,1) 
    pi[2] ~ dbeta(9,1) #prior probability of exclusion
    # prior mean of beta coefficient priors
    for(k in 1:2){
      mub[k] ~ dnorm(0,1)
      taub[k] ~ dt(0, 1, 1)  T(0,) # half cauchy prior 
    }
    # beta coefficient priors (spike and slab)- should generalize to less severe shrinkage (eg stochastic search variable selection)
    b0 ~ dnorm(0, 10)
    for(j in 1:2){ 
      delta[j] ~ dbern(pi[1])
      bpr[j] ~ dnorm(mub[1], taub[1])
      b[j] <- bpr[j]*(1-delta[j]) 
    }
    for(j in 3:14){
      delta[j] ~ dbern(pi[2])
      bpr[j] ~ dnorm(mub[2], taub[2])
      b[j] <- bpr[j]*(1-delta[j])
    }
  }
  "
tf = tempfile()
cat(jagsmod, file=tf)


analyze <- function(i=1, n=100, outfile, append=FALSE){
  data = dgm(n)
  jdat = as.list(data$obs)
  jdat$xint0 = data$int0$x
  jdat$xint1 = data$int1$x
  jdat$N = length(jdat$y)
  
  res <- jags.parallel(data=jdat, parameters.to.save=c("md", "m1", "m0"), model.file = tf,
                n.chains = 4, n.iter = 20000, n.burnin = 500,
                n.thin = 2, DIC = FALSE, jags.seed = NULL
  )
  
  m1t=mean(data$int1$y)
  m0t=mean(data$int0$y)
  re = apply(res$BUGSoutput$sims.matrix[,c("m1", "m0", "md")], 2, function(x) c(mean=mean(x), sd=sd(x), low=quantile(x, .025), up=quantile(x, .975)))
  res = c(m1t=m1t, m0t=m0t, mdt=m1t-m0t,
    as.numeric(t(re))
  )
  names(res) <- c("m1t","m0t","mdt","m1","m0","md","s1","s0","sd","m1l","m0l","mdl","m1u","m0u","mdu")
  write.table(as.data.frame(t(res)), file=outfile, append=append, sep=",", row.names = FALSE, quote=FALSE, col.names=ifelse(append, FALSE, TRUE))
  res
}





outfile = "n1000sel.csv"
nr <- c("m1t","m0t","mdt","m1","m0","md","s1","s0","sd","m1l","m0l","mdl","m1u","m0u","mdu")
write.table(as.data.frame(rbind(nr)), file=outfile, append=FALSE, sep=",", row.names = FALSE, quote=FALSE, col.names=FALSE)
plan(multiprocess)
nbrOfWorkers()


future_sapply(1:1000, analyze, n=1000, outfile=outfile, append=TRUE)


