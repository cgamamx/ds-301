# Title     : Simulation of HW 1 - Problem 4
# Created by: Cesar Gama
# Created on: 1/24/2021


simulate_experiment <- function(p=0.02, n=10000) {
  # The simulation will generate a sample of individuals, indicating if they have a genetic disorder, then will simulate
  # a test to detect if they have the disorder, and finally will compute the probability of the subjects having the
  # disorder based on the test results.
  #
  # Parameters:
  #     p (double): Probability of an individual within the population of having the genetic disorder
  #     n (double): Sample size | Number of hypotetical subjects
  # Returns:
  #     double:

  # For the simulation we define two events, A and B:
  #     Let A = "Test result is positive"
  #     Let B = "Has a genetic Disorder"; P(B) = parameter p (default to 0.02)
  B <- sample(c(0, 1), n,prob=c(1-p, p), replace=TRUE)
  A <- apply_test(B)

  # Then the empirical frequency of disease given positive is
  n_BA <- sum(A & B)
  n_A <- sum(A)
  pB_given_A <- n_BA / n_A
  return(pB_given_A)
}


apply_test <- function (subjects, sens=0.999, spec=0.995){
  # Simulates a test to a sample of subjects. Sensitibity and specifity could be passed in as parameters.
  #
  # It creates a vector u of size n, with random numbers uniformly distributed between {0, 1}. For each subject, we
  # compare u_i either against the sensitiv if it has ths disroder, or against the specifity if it doesn't have the
  # disorder.
  #
  # Parameters:
  #     B (vector): Sample of subjects to apply the test to. Values in {0, 1}
  #     sens (double): probability that the test returns a positive result, if the individual has the disorder
  #     spec (double): probability that the test reurns a negaative result, if the individual doesn't have the disorder
  #                 also defined as 1 - (false positive rate)
  #
  # Returns:
  #     vector: Vector with tests results. Values in {0, 1}

  n <- length(subjects)
  u <- runif(n)

  return(ifelse(subjects==1, u<sens, u<1-spec))
}


main <- function (){
  # Number of iterations
  n <- 1000

  # We want to compare how Pr(B|A) changes for a range of values of P(B)
  p_range <- seq(from=0.01,to=0.5,length=n)
  y <- rep(0,n)
  for(i in 1:n)
    y[i] <- simulate_experiment(p_range[i])

  # Finally, plot the results
  plot(p_range,y,type="l",bty="l",xlab="Pr(B)",ylab="Pr(B | A)")
  sens <- 0.999
  abline(h=sens,lty=2,col=2)
}


main()