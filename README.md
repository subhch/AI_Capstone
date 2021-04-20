# ibm-ai-workflow-submission


## 0. Runing the application 

To start this application run the following command:

`python app.py`

To run all the tests
 `python run-tests.py`

### 1) Assimilate the business scenario and articulate testable hypotheses.

Manager challenges : 

- Spending too much time in predicting revenue
- Predictions are not as accurate as they would like.

Goal:

- Create a service that, at any point in time, will predict the revenue for the following month.
- They have also asked that the service be given the ability to project revenue for a specific country.

Advice:

- Keep the development time reasonable you have been advised to limit your model to the ten countries with the most revenue.

Null hypothesis : 

Our service results are not more accurate than the current models of the managers.

### 2) State the ideal data to address the business opportunity and clarify the rationale for needing specific data

The idea data to address this business problem is to have :

- Historical data with monthly revenue
- Customer information (subscriber)
- Subscription type for each customer
- Country

### 3) Extract Relevant Data

- times_viewed
- price
- country
------
## EDA

Can be found  [Data_Ingestion & EDA Notebook](https://github.com/mouadzeghraoui/coursera-ibm-ai-workflow-submission/blob/main/Final_Capstone/notebooks/Part1%20Data%20ingestion%20%26%20EDA.ipynb)

---- 
# What is being evaluated?

Upon completion of this capstone you will need to submit your work for peer review.  The following questions are being evaluated as part of the peer review submission:

1. Are there unit tests for the API?
2. Are there unit tests for the model?
3. Are there unit tests for the logging?
4. Can all of the unit tests be run with a single script and do all of the unit tests pass?
5. Is there a mechanism to monitor performance?
6. Was there an attempt to isolate the read/write unit tests from production models and logs?
7. Does the API work as expected? For example, can you get predictions for a specific country as well as for all countries combined?
8. Does the data ingestion exists as a function or script to facilitate automation?
9. Were multiple models compared?
10. Did the EDA investigation use visualizations?
11. Is everything containerized within a working Docker image?
12. Did they use a visualization to compare their model to the baseline model?


