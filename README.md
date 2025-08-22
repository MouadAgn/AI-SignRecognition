À quoi servent les fichiers src/ donnés

src/demo_realtime.py
→ Vision en direct : affiche les landmarks de la main (21 points), la main gauche/droite, la bbox, les FPS.
→ C’est la brique de base pour valider que Mediapipe fonctionne (caméra, perf, stabilité).

src/demo_gestures_rules.py (Script 2 ci-dessus)
→ Démo “intelligente” sans entraînement : classe quelques gestes courants via des règles (distances/positions).
→ Idéal pour une présentation rapide (“notre système reconnaît déjà des gestes simples en live”).

src/train_asl_multiclass.py
→ Entraînement supervisé sur Sign Language MNIST (alphabet ASL en images 28x28).
→ Produit des métriques, matrice de confusion, PR-curves, et sauvegarde le meilleur modèle (outputs/models/asl_cnn.pt).
→ Coche les items pédagogiques du prof : early stopping, class weights / sampler, scheduler, PR-AUC.

src/utils_metrics.py
→ Fonctions utilitaires pour :

sauvegarder les logs (jsonl),

tracer la matrice de confusion,

tracer les courbes précision–rappel (macro AP),
→ Utilisé par train_asl_multiclass.py pour générer des figures prêtes à mettre dans les slides.
