R, Julia, and JAGS code to accompany the paper "Bayesian G-Computation to Estimate Impacts of Interventions on Exposure Mixtures: Demonstration with Metals from Coal-fired Power Plants and Birthweight"


File manifest:

- Julia programs (v. 1.3.0)
  - Gibbs.jl: Julia functions to complete a Bayesian hierarchical modeling analysis of simulated data using Hierarchical models within a Bayesian g-computation framework using Gibbs sampling based MCMC methods
  - sims_bayes.jl: a Julia program to perform parallel Bayesian analysis of multiple simulated datasets using the functions from Gibbs.jl
  -  sims_ml.jl: a Julia program to perform parallel maximum likelhood analysis of multiple simulated datasets using the GLM package in Julia


- R program (v. 3.6.0)
    - sims_bma.R: an R program to perform parallel Bayesian analysis of multiple simulated datasets using Bayesian model averaging within a Bayesian g-computation framework

- JAGS programs (v 4.3.0)
  - No files included, but JAGS programs are contained as character strings in sims_bma.R
