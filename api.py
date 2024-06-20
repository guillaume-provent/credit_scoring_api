import os
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

THRESHOLD = 0.6205400652376516

# Répertoire du fichier api.py :
current_directory = os.path.dirname(os.path.abspath(__file__))

# Chargement du modèle :
model_path = os.path.join(current_directory, "model.pkl")
model = joblib.load(model_path)

# Chargement des données :
data_path = os.path.join(current_directory, "data_api_medium.csv")
data = pd.read_csv(data_path)


@app.route('/predict', methods=['POST'])
def predict():
    # Récupération de l'identifiant du client à partir de la requête :
    sk_id_curr = request.json.get('SK_ID_CURR')

    # Vérification de la présence de l'identifiant dans les données :
    if sk_id_curr not in data['SK_ID_CURR'].values:
        return jsonify({'error': 'Identifiant non trouvé'}), 404

    # Sélection de la ligne correspondant à SK_ID_CURR :
    row = data[data['SK_ID_CURR'] == sk_id_curr].drop(columns=['SK_ID_CURR'])

    # Prédiction :
    proba = model.predict_proba(row)[0, 1]
    prediction = (proba > THRESHOLD).astype(int)

    # Verbalisation de la prédiction :
    if prediction == 0:
        result = 'DOSSIER ACCEPTE'
    else:
        result = 'DOSSIER REFUSE'

    # Renvoi du résultat :
    return jsonify({'Identifiant': sk_id_curr, 'Dossier': result, 'Probabilite': proba, 'Seuil': THRESHOLD})


if __name__ == '__main__':
    app.run(debug=True)
