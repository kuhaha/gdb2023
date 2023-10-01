## import dependencies
from sklearn import tree #For Decision Tree
import pandas as pd # For DataFrame
import pydotplus # To create Decision Tree Graph

### - 1 - ###
#Read the dataset
golf_df=pd.read_csv('golf_play.csv',dtype='category')

#Print/show the new data
print(golf_df)

### - 2 - ###
# Convert categorical variable into dummy/indicator variables or (binary vairbles) essentialy 1's and 0's
one_hot_data = pd.get_dummies(golf_df[ ['Outlook', 'Temperature', 'Humidity', 'Windy'] ])
#print the new dummy data
print(one_hot_data)

### - 3 - ###
# The decision tree classifier. criterion="entropy", default criterion="gini"
clf = tree.DecisionTreeClassifier() 
# Training the Decision Tree
clf_train = clf.fit(one_hot_data, golf_df['Play'])

### - 4 - ###
# Test model prediction input:
print ("Outlook=sunny, Temperature=hot, Humidity=normal, Windy=false")
case = pd.DataFrame([[0,0,1,0,1,0,0,1,1,0]],columns=one_hot_data.columns.values)
prediction = clf_train.predict(case)
print("predictions : " + str(prediction))

### -5- ###
# Export/Print a decision tree in DOT format.
print(tree.export_graphviz(clf_train, None))

#Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(one_hot_data.columns.values), 
                                class_names=['Not_Play', 'Play'], rounded=True, filled=True) 

#Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Windows上ではGraphvizのインスト―ル＋以下の追加部分(インストール先によって適宜修正)
#graph.progs = {'dot': u"D:\\local\\Graphviz\\bin\\dot.exe"}  

# Output graph
graph.write_png('golf_play.png')