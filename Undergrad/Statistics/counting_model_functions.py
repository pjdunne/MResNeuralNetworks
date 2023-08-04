# Lot's of Functions For our counting experiment
import numpy
from counting_model import *

# Poisson mean
def lamb(mu,eta):
  return mu*eff*A*l*((1+k)**eta)*sigma_TH + B

# Functions for Frequentist calculations -> 

# define q and its derivatives wrt eta
def q(mu,eta,np,eta_p):
  la = lamb(mu,eta)
  if np==0: return (eta-eta_p)*(eta-eta_p) + 2*la
  elif la<=0: return 99999
  else: return (eta-eta_p)*(eta-eta_p) + 2*la -2*np*numpy.log(la)

# and a handy format for optimize packages
def q_formin(x,args):
    vmu=x[0]
    veta=x[1]
    vn=args[0]
    veta_p=args[1]
    return  q(vmu,veta,vn,veta_p)

# dq/deta
def dq(n_p,eta_p,mu,eta):
  la = lamb(mu,eta)
  dl = (la-B)*numpy.log(1+k)
  return 2.*(eta-eta_p) + 2.*dl - 2.*n_p*dl/la

# d^2q/deta^2
def d2q(n_p,mu,eta):
  log_k = numpy.log(1+k)
  la    = lamb(mu,eta)
  dl    = log_k*(la-B)
  d2l   = log_k*log_k*(la-B)
  return 2. + 2.*d2l - 2.*n_p*(1./(la*la))*dl*dl - 2.*n_p*(1./la)*d2l

# Newton method to find profiled eta value at fixed mu
def profiled_eta(n_p,eta_p,mu):
  # numerical minimumisation via newton method
  tol = 0.01
  init_eta = -3
  eps = 100
  while eps > tol:
    qp = dq(n_p,eta_p,mu,init_eta)
    eps = abs(qp)
    init_eta = init_eta - qp/d2q(n_p,mu,init_eta)
  return init_eta

# Functions for Bayesian calculations 

# Likelihood --> P(n|mu,eta)
def Likelihood(n,mu,eta):
  l = lamb(mu,eta)
  return (l**n)*numpy.exp(-l)/numpy.math.factorial(int(n))

# prior on eta P(eta)
def prior_eta(eta): 
  c = 1./((2*numpy.pi)**0.5)
  return c*numpy.exp(-0.5*eta*eta)

# prior on mu P(mu)
def prior_mu(mu):
  return 1./25 if (mu < 25 and mu > 0) else 0 

# and define the product of them (the numerator in Bayes rule)
def product(eta,mu,n):
  return Likelihood(n,mu,eta)*prior_mu(mu)*prior_eta(eta)

import scipy.integrate as integrate

# integrate over eta
def integral(mu,n):
  return integrate.quad(product,-6,6,args=(mu,n),epsabs=1.49e-03)[0]
    
# and then integrate over mu
def norm(n):
  return integrate.dblquad(product,0.01,25,lambda x:-6,lambda x:6,args=[n],epsabs=1.49e-03)[0]