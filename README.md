# Celonis-21-Challenge

Our team was tasked with identifying the underlying cause of the underperforming Pizzeria Mamma Mia. The two important metrics that we identified are the net gain of each sale and the customer satisfaction score. In order to study the relationship between the variables that were provided to our team and the two metrics  and due to the time constraints, we decided to use a random forest algorithm to synthesize a predictive model.

All our models were run on Python using the sklearn libraries. Due to the nature of the values of the variables, we replaced non-numerical values with a numeric label and ran our model using a separate label, one being the net gain and the other being the customer satisfaction level. With 5000 trees, we achieved an accuracy level of 40% when predicting the customer satisfaction classification and an accuracy level of 98% when predicting the net gain made from one purchase. If we have more time, we would deploy multi-class random forest algorithm in order to handle both class at the same time which would represent a better realistic prediction.

Youtube link:https://youtu.be/DIbF4youLPw
