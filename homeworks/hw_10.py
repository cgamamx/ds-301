import pandas as pd
from dsstats.inference import samples
from dsstats.inference import hypothesis

immune_tea_data = pd.read_csv('../dsstats/data/ImmuneTea.csv')

# Q11. The manufacturers are interested in estimating the percentage of defective light bulbs coming from a certain
# process. They want a 90% confidence interval with a margin of error of 2%.  How many light bulbs must they test?
sample_size = samples.single_proportion_sample_size(margin=0.02, confidence_interval=0.90)
print(sample_size)

# Q12. Same question as in the previous problem, but assume they had a reason to believe the proportion is fairly close
# to 6%. How large a sample must they test?
sample_size = samples.single_proportion_sample_size(p_tilde=0.06, margin=0.02, confidence_interval=0.90)
print(sample_size)

# Q13. An airline has a regular flight between two cities.  From a previous study, we estimate the standard deviation of
# the flight times to be 9.34 minutes.  We want a 99% confidence interval for the average flight time  with a margin of
# error of 3minutes. How large a sample would we need to find that confidence interval?
sample_size = samples.single_mean_sample_size(sigma_tilde=9.34, margin=3, confidence_interval=0.99)
print(sample_size)

# Q14. NOOP

# Q15. Data: Immune Tea, Test:
interferon_gamma = immune_tea_data.groupby('Drink').describe().get('InterferonGamma').transpose()
test_result = hypothesis.two_mean_test(interferon_gamma, ('Tea', 'Coffee'), tail='right')
print(test_result)
