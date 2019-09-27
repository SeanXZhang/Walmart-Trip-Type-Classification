# Walmart Trip Type Classification
## [ Overview ]
### (1) Team Members : 
> #### Sean Zhang
> #### Zheng Xu
### (2) Dataset :
> #### Walmart Shopping Records

### (3) Objective :
> #### Classification of 38 Trip types of each customer based on thier shopping data (e.g. department, Upc, FinelineNumber, Weekday, Scan count) 
<br>

## [Data Description]

> #### train : 647054 rows, 7 columns
> #### test : 653646 rows, 6 columns
> - identical features except the target value (TripType)

| Index | Feature               | Feature Description                                  | Unique Value |
|-------|-----------------------|----------------------------------------------|--------|
| 1     | TripType              | A categorical id representing the type of shopping trip the customer made.                                       | 38     |
| 2     | VisitNumber           | An id corresponding to a single trip by a single customer                              | 95674  |
| 3     | Weekday               | The weekday of the trip                    | 7      |
| 4     | Upc                   | The UPC number of the product purchased                  | 97715  |
| 5     | ScanCount             | The number of the given item that was purchased. A negative value indicates a product return          | 39     |
| 6     | DepartmentDescription | A high-level description of the item's department                                | 69     |
| 7     | FinelineNumber        | A more refined category for each of the products, created by Walmart | 5196   |


<br>

## [Evaluation] : Multi-class log loss (Cross Entropy)
> ![](https://github.com/yunah0515/dss7_SWYA_walmart/blob/master/image/evaluation.png?raw=true)
<br>

## [Contents]

### (1) Challenges
> - Each observation represented an item rather than a visit. 
> - Need to group observations by visit to classify the trip.
> - Uneven Distribution of TripType.
> - Records with incomplete values, however many of them do contain some information.
> - Dummy variables (Categorical) - Weekday; Converted qualitative values to quantitative. Eg: Monday =1, Tuesday =2.
> - Duplicate department labels. Eg: “MENSWEAR” and“MENS WEAR”.
> - Number of unique UPCs and Fineline Numbers are large and without direct meanings, either need to select the top most frequent UPC and Fineline number categories and decoding are required.


### (2) Feature Engineering
> - Removing Missing Values
> - ScanCount seperation
> - Encoding Weekday, Department Descriptions
> - Dummy variables
> - Identifing the most frequently purchased items per VisitNumber

### (3) Modeling
> - Ensemble Methods:
Random Forest Classifier
Extreme Gradient Boost Classifier (XGBoost)
> - Neural Networks:
Keras classifiers with 1, 2, 3 and 4 hidden layers
> - Stacking Classifiers:
Neural networks as both the first-level classifiers and meta-classifier
Neural networks based stacking with multinomial logistic regression as meta-classifier
Neural networks based stacking with XGBoost as meta-classifier


### (4) Results
> - Logarithmic loss : 0.78259
> - Accuracy score : 73.68%
> - Feature Importance Top 20

