import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
train_df=pd.read_csv("train.csv")
print("---- VERİ SETİ BİLGİSİ ---")
train_df.info()

print("\n--- EKSİK VERİ SAYILARI---")
print(train_df.isnull().sum())

train_df = train_df.drop(columns=['Cabin'])
train_df['Age']= train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked']=train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

print("\n--- TEMİZLİK SONRASI EKSİK VERİ DURUMU -----")
print(train_df.isnull().sum())

train_df=train_df.drop(columns=['PassengerId','Name','Ticket'])
train_df['Sex']=train_df['Sex'].map({'male':0,'female':1})

train_df=pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)
print("\n---- SAYISALLAŞTIRILMIŞ VERİ SETİ ----")
print(train_df.head())

X = train_df.drop(columns=['Survived'])
y=train_df['Survived']
X_train, X_test ,y_train, y_test =train_test_split(X,y, test_size=0.2, random_state=42)
model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\n--- MODEL EĞİTİMİ TAMAMLANDI ---")

tahminler=model.predict(X_test)
dogruluk_orani = accuracy_score(y_test, tahminler)
print(f"\n--- MODELİN KARNESİ ----")
print(f"\nYapay Zekanın Doğruluk Oranı: %{dogruluk_orani *100:.2f}")
