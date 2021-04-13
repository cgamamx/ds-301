import pandas as pd
from dsstats.inference import hypothesis
from dsstats.inference import confidence

salary_data = pd.read_csv('dsstats/data/SalaryGender.csv')
student_data = pd.read_csv('dsstats/data/StudentSurvey.csv')
cocaine_data = pd.read_csv('dsstats/data/CocaineTreatment.csv')

# Q1. Data: Salary Gender, Test: Avg of college teachers is less than 50 years
test_result = hypothesis.single_mean_test(salary_data['Age'], mu_0=50, alternative='less')
print(f'Q1: The p-value is {test_result["p-value"]}')

# Q2. Data: Student Survey, CI: 90% for mean of Math SAT score
confidence_interval = confidence.single_mean_interval(student_data['MathSAT'], ci=0.90)
print(f'Q2: The length of the confidence interval is {confidence_interval[0]-confidence_interval[1]}')

# Q3: NOOP
print('Q3: NOOP')

# Q4. Data: Cocaine Treatment, Test: Among those who received placebo, the proportion of non-relapse is less than 0.3
p_relapse_with_placebo = (cocaine_data.where(cocaine_data['Drug'] == 'Placebo')
                                      .groupby('Relapse', dropna=True).count()
                                      .squeeze())
test_result = hypothesis.single_proportion_test(p_relapse_with_placebo, category='no', p_0=0.30,
                                                alternative='less')
print(f'Q4: The p-value is {test_result["p-value"]}')

# Q5. Data: Cocaine Treatment, Test: The proportion of non-relapse is higher for those treated with Lithium than placebo
p_non_relapse = (cocaine_data.where(cocaine_data['Relapse'] == 'no')
                             .groupby('Drug', dropna=True).count()
                             .squeeze())
test_result = hypothesis.two_proportions_test(p_non_relapse, categories=('Lithium', 'Placebo'),
                                              alternative='greater')
print(f'Q5: The p-value is {test_result["p-value"]}')

# Q6. Data: Cocaine Treatment, CI: 83% for the difference of proportion in the population who are expected to have no
# relapse, between those treated with Desipramine and those treated with Lithium
confidence_interval = confidence.two_proportions_interval(p_non_relapse, categories=('Desipramine', 'Lithium'), ci=0.83)
print(f'Q6: The length of the confidence interval is {confidence_interval[0]-confidence_interval[1]}')