import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn import svm

data = pd.read_csv('diabetes.csv')

# label = 'Outcome'

# # Calculate correlation matrix
# correlation_matrix = data.corr()

# # Extract correlations with respect to the label attribute
# correlations_with_label = correlation_matrix[label]

# # Display the correlations
# print("Correlation of each feature with respect to the label:")
# print(correlations_with_label)




X = data.drop(['Outcome', 'SkinThickness', 'BloodPressure'], axis=1)
y = data['Outcome']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)




scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

model = svm.SVC(kernel='linear')



model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(pred, y_test)
print("Accuracy:", accuracy)



input_data = (8,s183,0,23.3,0.672,32)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
input_std = scaler.transform(input_data_reshaped)
pred = model.predict(input_std)
print(pred)

if (pred[0]==0):
    print("The person is not diabetic")

else:
    print("The person is diabetic")

