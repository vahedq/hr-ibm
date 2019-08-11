import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns

from statannot import add_stat_annotation
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Bidirectional
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.python.keras.layers import LSTM, Flatten
from tensorflow.python.keras.layers import Reshape, Conv1D, MaxPooling1D



df = pd.pandas.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv', index_col=None)
front = df['Attrition']
df.drop(labels=['Attrition'], axis=1, inplace=True)
df.insert(0, 'Attrition', front)
df.head()
df.shape

# Set up the matplotlib figure
f, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 6))

def sub_bar_plot(predictor, title, ax):
    sns.barplot(
        x=predictor, y=predictor, hue='Attrition',
        data=df, ax=ax, palette='husl',
        estimator=lambda x: len(x) / len(df) * 100
    ).set_title(title)
    percent = ax.set(ylabel='Percent')
    percent = ax.set(xlabel='')


sub_bar_plot('YearsInCurrentRole', 'Uears in Current Role', axes[0][0])
sub_bar_plot('YearsSinceLastPromotion', 'Years from Last Promotion', axes[0][1])
sub_bar_plot('YearsAtCompany', 'Years at Company', axes[1][0])
sub_bar_plot('TotalWorkingYears', 'Total Working Years', axes[1][1])
plt.subplots_adjust(hspace=0.52)

corr = df.corr()
corr = (corr)
plt.figure(figsize=(18,8))
sns.heatmap(corr,
            square=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap=sns.cm.rocket_r)
plt.title('Heatmap of Correlation Matrix')


sns.barplot(x='Attrition', y='DistanceFromHome', hue='Gender', data=df*1, palette='husl')
plt.show()

plt.figure(figsize=(10,6))

x='EducationField'
y='YearsAtCompany'
hue='Gender'
ax = sns.boxplot(x=x, y=y, hue=hue, data=df, palette='PRGn')
add_stat_annotation(
    ax, data=df, x=x, y=y, hue=hue,
    boxPairList=[
        (('Life Sciences', 'Female'), ('Life Sciences', 'Male')),
        (('Other', 'Female'), ('Other', 'Male')),
        (('Medical', 'Female'), ('Medical', 'Male')),
        (('Technical Degree', 'Female'), ('Technical Degree', 'Male')),
        (('Human Resources', 'Female'), ('Human Resources', 'Male')),
        (('Marketing', 'Female'), ('Marketing', 'Male')),
    ],
    test='t-test_ind', textFormat='star', loc='inside', verbose=2)

plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
plt.xticks(rotation=-90)


df_cat = df[['Attrition', 'BusinessTravel','Department',
             'EducationField','Gender','JobRole',
             'MaritalStatus',
             'Over18', 'OverTime']].copy()
num_val = {'Yes':1, 'No':0}
df_cat['Attrition'] = df_cat['Attrition'].apply(lambda x: num_val[x])
df_cat = pd.get_dummies(df_cat)

df_num = df[[
    'Age','DailyRate','DistanceFromHome', 
    'EnvironmentSatisfaction', 'HourlyRate',                     
    'JobInvolvement', 'JobLevel',
    'JobSatisfaction',
    'MonthlyIncome',
    'MonthlyRate',
    'RelationshipSatisfaction', 
    'StockOptionLevel',
    'TrainingTimesLastYear',
    'TotalWorkingYears',
    'WorkLifeBalance',
    'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]]
df_final = pd.concat([df_cat, df_num], axis=1)
df_final.head()

y = df_final['Attrition']
X = df_final.drop('Attrition', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

input_dim = X_train.shape[1]

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

model = Sequential()
model.add(Conv1D(16, kernel_size=3, input_shape=(X_train.shape[1],1), activation='sigmoid'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(8, kernel_size=3, activation='sigmoid'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(4, kernel_size=3, activation='sigmoid'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train,
    y_train,
    epochs=2000,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)

y_pred = model.predict_classes(X_test)
print('Classification Report:')
print(classification_report(y_test, y_pred))