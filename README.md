# API de scoring crédit

Ce repository contient l'API destinée à la prédiction du risque de défaut de remboursement d'un prêt par un client.
A partir d'une requête (/predict) avec l'indenfifiant du client ('SK_ID_CURR'), l'API renvoie la mention "DOSSIER ACCEPTE" ou "DOSSIER REFUSE".

Le fichier data_api_medium.csv contient les données nécessaires à la prédiction, et le ficher model.pkl le modèle de prédicition.
En cas de mise à jour du modèle, la constante THRESHOLD (seuil de classification) dans le fichier api.py devra être modifiée selon le paramétrage du modèle. 

Le repository est paramétré avec Github Actions pour un déploiement continu sur Heroku, incluant les tests unitaires du fichier test_api.py.