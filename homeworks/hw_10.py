import pandas as pd
from statistics.inference import samples
from statistics.inference import hypothesis

immune_tea_data = pd.read_csv('statistics/data/ImmuneTea.csv')
wetsuit_data = pd.read_csv('statistics/data/Wetsuits.csv')

# Q7.a Data: Wetsuits (difference of two means), Test: is there a difference in swimming speeds due to wearing a wetsuit
test_result = hypothesis.two_mean_test(wetsuit_data.describe(), ('Wetsuit', 'NoWetsuit'),
                                       alternative='two-sided')
print(f'Q7: The p-value of the difference of means is {test_result["p-value"]}')

# Q7.a Data: Wetsuits (matched pair), Test: is there a difference in swimming speeds due to wearing a wetsuit
wetsuit_data_difference = wetsuit_data['NoWetsuit'] - wetsuit_data['Wetsuit']
test_result = hypothesis.single_mean_test(wetsuit_data_difference, mu_0=0, alternative='two-sided')
print(f'Q7: The p-value as matched pairs is {test_result["p-value"]}')

# Q7.b
# Q7.c Data Manipulation for the StatKey Simulation
wetsuit_statkey = pd.DataFrame(
                                data={
                                    'Time': wetsuit_data['Wetsuit'],
                                    'Wetsuit': 'yes'
                                })
wetsuit_statkey = wetsuit_statkey.append(pd.DataFrame(
                                                        data={
                                                            'Time': wetsuit_data['NoWetsuit'],
                                                            'Wetsuit': 'no'
                                                        }
                                                      ), ignore_index=True)
# File for Two Means Simulation
wetsuit_statkey.to_csv('/tmp/wetsuit_statkey_two-means.csv', index=False)

# File for matched pairs Simulation
wetsuit_data['Difference'] = wetsuit_data['Wetsuit'] - wetsuit_data['NoWetsuit']
wetsuit_data.to_csv('/tmp/wetsuit_statkey_matched-pairs.csv', index=False)

# Q11. The manufacturers are interested in estimating the percentage of defective light bulbs coming from a certain
# process. They want a 90% confidence interval with a margin of error of 2%.  How many light bulbs must they test?
sample_size = samples.single_proportion_sample_size(margin=0.02, confidence_interval=0.90)
print(f'Q11: The size of the sample needed is {sample_size}')

# Q12. Same question as in the previous problem, but assume they had a reason to believe the proportion is fairly close
# to 6%. How large a sample must they test?
sample_size = samples.single_proportion_sample_size(p_tilde=0.06, margin=0.02, confidence_interval=0.90)
print(f'Q12: The size of the sample needed is {sample_size}')

# Q13. An airline has a regular flight between two cities.  From a previous study, we estimate the standard deviation of
# the flight times to be 9.34 minutes.  We want a 99% confidence interval for the average flight time  with a margin of
# error of 3minutes. How large a sample would we need to find that confidence interval?
sample_size = samples.single_mean_sample_size(sigma_tilde=9.34, margin=3, confidence_interval=0.99)
print(f'Q13: The size of the sample needed is {sample_size}')

# Q14. NOOP
print('Q14: NOOP')

# Q15. Data: Immune Tea, Test: Production of interferon gamma is enhanced in tea drinkers?
interferon_gamma = immune_tea_data.groupby('Drink').describe().get('InterferonGamma').transpose()
test_result = hypothesis.two_mean_test(interferon_gamma, ('Tea', 'Coffee'), alternative='greater')
print(f'Q15: The p-value is {test_result["p-value"]}')
test_result = hypothesis.two_mean_test(interferon_gamma, ('Tea', 'Coffee'), alternative='greater', df='satterthwait')
print(f'Q15: The p-value, using the Satterthwait approximation is {test_result["p-value"]}')
