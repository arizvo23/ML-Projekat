import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Učitavanje podataka iz CSV datoteke 1.csv
df1 = pd.read_csv('1.csv')

# Dodavanje ciljne varijable 'stanje_na_putu' u DataFrame
df1['stanje_na_putu'] = pd.Series(['OK' if x == 'ok' else 'Not OK' for x in df1['stanje_na_putu']], index=df1.index)

# Učitavanje podataka iz CSV datoteke 2.csv
df2 = pd.read_csv('2.csv')

# Dodavanje ciljne varijable 'stanje_na_putu' u DataFrame
df2['stanje_na_putu'] = pd.Series(['OK' if x == 'ok' else 'Not OK' for x in df2['stanje_na_putu']], index=df2.index)

# Spajanje DataFrame-ova
df = pd.concat([df1, df2])

# Odabir značajki
features = ['acc_z']

# Definiranje ciljne varijable
target = 'stanje_na_putu'

# Podjela skupa podataka na skupove za treniranje i testiranje
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Definiranje modela
model = RandomForestClassifier()

# Treniranje modela
model.fit(X_train, y_train)

# Predviđanje ciljne varijable na testnom skupu
y_pred = model.predict(X_test)

# Izračun točnosti modela
accuracy = accuracy_score(y_test, y_pred)
print('Točnost modela: {:.2f}%'.format(accuracy * 100))

# Spremanje modela u datoteku
import pickle
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)

# Učitavanje modela iz datoteke
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)
