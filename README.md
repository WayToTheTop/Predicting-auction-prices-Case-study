Regression Case Study
======================

This is the model built by my team during Regression Case Study in Galvanize DSI program.

The goal of the contest was to predict the sale price of a particular piece of
heavy equipment at auction based on it's usage, equipment type, and
configuration.  The data is sourced from auction result postings and includes information on usage and equipment configurations.

Evaluation
======================
The evaluation of the model was based on Root Mean Squared Log Error.
Which is computed as follows:

![Root Mean Squared Logarithmic Error](images/rmsle.png)

where *p<sub>i</sub>* are the predicted values and *a<sub>i</sub>* are the
target values.

Note that this loss function is sensitive to the *ratio* of predicted values to
the actual values, a prediction of 200 for an actual value of 100 contributes
approximately the same amount to the loss as a prediction of 2000 for an actual value of 1000.   

This loss function is implemented in `score_model.py`.

Restrictions
============
We were restricted to using *regression* methods only for this case study.  
The following techniques were legal

  - Linear Regression.
  - Logistic Regression.
  - Median Regression (linear regression by minimizing the sum of absolute deviations).
  - Any other GLM.
  - Regularization: Ridge and LASSO.

Results
=============
Having only 4 hours our to work on a problem, our team built a model that reduced
baseline error by 16.5%. Our team got 2nd best score loosing only slightly to
the best model.

On the base of our model we conclude that the main predictors for auction price are
product group of the equipment, its age and the place of the auction (we used auctioneer ID as a proxy for location)

The largest challenge of the project was messy and sparse data.  We spent lots of
time carefully engineering useful features and considering ways to deal with
absent data.

Future working
===============
I think that the most promising direction for future work is to extract from
data average sale price for the last few transactions for each equipment model.
This information was not given explicitly but might be extracted from data.

Description
===========
`run.py` trains linear model on the data in `data/Train.csv` and makes a prediction for `data/test.csv`. Predictions are saved to file `predictions.csv`.

All functions are in `ridge_model.py`.   
