# Walmart Trip Type Classification
## [ Overview ]
### (1) Team Members : 
> #### Sean Zhang
> #### Zheng Xu
### (2) Dataset :
> #### Walmart Shopping Records

### (3) Objective :
> #### Classification of 38 Trip types of each customer based on thier shopping data
(e.g. department, Upc, FinelineNumber, Weekday, Scan count) 
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
> - Missing Values
> - Encoding Weekday
> - Uneven Distribution of TripType
> - Most Frequent & Least Frequent TripType

### (2) Feature Engineering
> - Removing Missing Values
> - ScanCount seperation
> - Encoding Weekday, Department Descriptions
> - Dummy variables
> - Identifing the most frequently purchased items per VisitNumber

### (3) Modeling
> - RandomForest
> - XGBoost

### (4) Results
> - Logarithmic loss : 0.78259
> - Accuracy score : 73.68%
> - Feature Importance Top 20

