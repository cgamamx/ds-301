# Title     : Simulation of HW 1 - Problem 4
# Objective : See how changing Pr(B) impacts Pr(B|A)
# Created by: Cesar Gama
# Created on: 1/24/2021

# Let A = "positive test"
# Let B = "Genetic Disorder"

# Generate n hypotetical subjects, generating indicators Bi in {0, 1}, i=1,2,3,..,n
# For subject i having the genetic disorder
f <- function(p=0.02) {
  n <- 1000000
  B <- sample(c(0, 1), n,prob=c(1-p, p), replace=TRUE)

  # Now we simulate test results for the n subjects
  sens <- 0.999
  spec <- 1 - 0.005
  u <- runif(n)
  A <- ifelse(B==1, u<sens, u<1-spec)

  # Then the empirical frequency of disease given positive is
  nBA <- sum(A & B)
  nA <- sum(A)
  pBgivenA <- nBA / nA
  return(pBgivenA)
}

# Finally, create a grid and plot it
pgrid <- seq(from=0.01,to=0.5,length=100)
y <- rep(0,100)
for(i in 1:100)
  y[i] <- f(pgrid[i])
plot(pgrid,y,type="l",bty="l",xlab="Pr(B)",ylab="Pr(B | A)", main='Probability that the person has the disorder if the test was positive, for different probabilities of B')
sens <- 0.999
abline(h=sens,lty=2,col=2)
