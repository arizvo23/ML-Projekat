import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier

# Učitavanje podataka iz CSV datoteke
data = pd.read_csv('3.csv')

# Odabir značajki
features = ['acc_z']

# Učitavanje treniranog modela
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

# Predviđanje ciljne varijable na novim podacima
y_pred = model.predict(data[features])

# Ispisivanje predviđenih vrijednosti ciljne varijable
print('Predviđena stanja na putu:')
print(y_pred)

# Dodavanje predviđenih vrijednosti u novu kolonu u DataFrameu
data['pred_stanje_na_putu'] = y_pred

# Spremanje novog DataFramea u CSV datoteku
data.to_csv('3.csv', index=False)