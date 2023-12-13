import os
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import model_selection
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("heart.csv")
sick_df = df[df['output'] == 1] # Hastaları ayır
healthy_df = df[df['output'] == 0] # Sağlıklıları ayır

num_examples = len(df)
num_sick = len(sick_df)
num_healthy = len(healthy_df)
print(f"Örnek Sayısı: {num_examples}")
print(f"Hasta Örnek Sayısı: {num_sick}")
print(f"Sağlıklı Örnek Sayısı: {num_healthy}")
feature_types = df.dtypes
print(feature_types)

#1.a.Ayırt edici özellik kontrolü
print(df.head()) #İlk 5 Satiri görmemizi sağliyor.
#Özelliklerin kontrolünü sağlamak için kullanilabilir

#1.b.Boş bırakılmış özellik için kontrol sağlanması
print(df.isnull().sum())

#1.c.Test ve Eğitim için verinin rastgele bölünmesi
X = df.drop('output', axis=1) #İnput değerlerinin ayrıştırılması
y = df['output'] #Output değerlerinin ayrıştırılması
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

#2.a.Hastaların yaş dağılınımın histogramı
sns.histplot(sick_df['age'],kde=False)
sns.set_style("whitegrid")
plt.title('Hastalarin Yaş Dağilimi Histogrami')
plt.xlabel('Yaş')
plt.ylabel('')
plt.show()

#2.b.Hasta ve sağlam olan kişilerin farklı 
#özellikler için dağılımını gösteren histogramlar
for i, col in enumerate(df.columns,1):
    fig=px.histogram(df,
                 x= df[col],
                 color="output",
                 hover_data=df.columns,
                 title=f"Distribution of {col} Data"
                )
    fig.show()

string_col=df.select_dtypes("string").columns.to_list()
df_nontree=pd.get_dummies(df,columns=string_col,drop_first=False)
print(df_nontree.head())

#3.Öğrenme Modeli
#Eğitim ve doğrulama için k-fold cross validation
k_fold = 5
X_train_val_splits = []
y_train_val_splits = []

for train_index, val_index in StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42).split(X_train_val, y_train_val):
    X_train_val_splits.append(X_train_val.iloc[train_index])
    y_train_val_splits.append(y_train_val.iloc[train_index])

#k-en yakın komşuluk, karar ağaçları, yapay nöron ağları
knn_model = KNeighborsClassifier(n_neighbors=13, metric='manhattan') 
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=5, min_samples_leaf=5)
mlp_model = MLPClassifier(hidden_layer_sizes=(25,), max_iter=500)

for i in range(k_fold):
    X_train = pd.concat([X_train_val_splits[j] for j in range(k_fold) if j != i], ignore_index=True)
    y_train = pd.concat([y_train_val_splits[j] for j in range(k_fold) if j != i], ignore_index=True)
    X_val = X_train_val_splits[i]
    y_val = y_train_val_splits[i]

    #k-en yakın komşuluk
    knn_model.fit(X_train, y_train)
    knn_score = knn_model.score(X_val, y_val)
    print(f'k-NN Model (Fold-{i+1}) Doğruluk Oranı: {knn_score}')

    #Karar ağaçları
    dt_model.fit(X_train, y_train)
    dt_score = dt_model.score(X_val, y_val)
    print(f'Decision Tree Model (Fold-{i+1}) Doğruluk Oranı: {dt_score}')

    #Yapay nöron ağları
    mlp_model.fit(X_train, y_train)
    mlp_score = mlp_model.score(X_val, y_val)
    print(f'MLP Model (Fold-{i+1}) Doğruluk Oranı: {mlp_score}')

#4.Test Aşaması
knn_test_score = knn_model.score(X_test, y_test)
dt_test_score = dt_model.score(X_test, y_test)
mlp_test_score = mlp_model.score(X_test, y_test)

print(f'\nTest Seti Doğruluk Oranları:')
print(f'k-NN Model Doğruluk Oranı: {knn_test_score}')
print(f'Decision Tree Model Doğruluk Oranı: {dt_test_score}')
print(f'MLP Model Doğruluk Oranı: {mlp_test_score}')
