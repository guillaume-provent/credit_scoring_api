import pytest
import joblib
import pandas as pd
from api import app, model_path, data_path


# Configuration de pytest pour utiliser le client de test de Flask
@pytest.fixture
def client():

    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# Test de chargement du modèle
def test_model_loading():

    model = joblib.load(model_path)
    assert model is not None


# Test de chargement des données
def test_data_loading():

    data = pd.read_csv(data_path)
    assert not data.empty
    assert 'SK_ID_CURR' in data.columns


# Test de la prédiction - Identifiant existant
def test_predict_valid_id(client):

    data = pd.read_csv(data_path)
    sk_id_curr = data['SK_ID_CURR'].iloc[0]
    sk_id_curr = int(sk_id_curr)

    response = client.post('/predict', json={'SK_ID_CURR': sk_id_curr})
    json_data = response.get_json()

    assert response.status_code == 200
    assert 'Identifiant' in json_data
    assert 'Dossier' in json_data
    assert 'Probabilite' in json_data
    assert 'Seuil' in json_data


# Test de la prédiction - Identifiant invalide
def test_predict_invalid_id(client):

    response = client.post('/predict', json={'SK_ID_CURR': 999999})
    json_data = response.get_json()

    assert response.status_code == 404
    assert 'error' in json_data
    assert json_data['error'] == 'Identifiant non trouvé'
