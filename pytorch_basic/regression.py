from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
print(X.shape, y.shape)
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size = 0.2)
print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)
#这里是线性回归模型，所以y_pred是连续值，需要round()到最近的整数
model = LinearRegression()
# model2 = LogisticRegression(multi_class = 'ovr')
model2 = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')

model.fit(train_X, train_y)
model2.fit(train_X, train_y)

y_pred = model.predict(test_X)
acc = accuracy_score(test_y, y_pred.round())
print("accuracy: ", acc)
y_pred2 = model2.predict(test_X)
acc2 = accuracy_score(test_y, y_pred2)
print("accuracy2: ", acc2)
print("mean squared error: ", mean_squared_error(test_y, y_pred))   
print("mean squared error2: ", mean_squared_error(test_y, y_pred2))   

print("R2 score: ", r2_score(test_y, y_pred))
print("R2 score2: ", r2_score(test_y, y_pred2))

print("the coefficient of determination is: ", model.score(test_X,test_y))
print("the coefficient of determination2 is: ", model2.score(test_X,test_y))

