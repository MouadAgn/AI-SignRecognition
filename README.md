AI-SignRecognition — README

Projet de reconnaissance de l’alphabet ASL en temps réel, avec transformation lettres → mots → phrases.
Deux voies complémentaires sont incluses :

Voie A (académique) : CNN entraîné sur Sign Language MNIST (images 28×28).

Voie B (démo live robuste – recommandée) : MediaPipe Hands (21 repères) → features invariantes → MLP scikit-learn + post-traitement (stabilisation, auto-correction de mots, phrase propre).
Cette voie résout le décalage « webcam réelle » vs « images parfaites 28×28 ».

Sommaire

Contexte et choix techniques

Fonctionnalités

Arborescence du dépôt

Pré-requis

Installation & environnement (uv)

Datasets & préparation

Sign Language MNIST (28×28)

ASL Alphabet (images réalistes) → extraction de landmarks

Voie A — CNN 28×28 (pour le rapport)

Voie B — Landmarks → MLP (pour la démo live)

Extraction auto des features

Entraînement MLP

Démo webcam — Lettres → Mots → Phrases

Options “radiographie” (fond sombre) et segmentation

Raccourcis clavier & UX

Dépannage (FAQ)

Limites connues

Contexte et choix techniques

Au départ, le projet utilisait Sign Language MNIST (images 28×28, fond propre). Les premiers résultats hors-ligne étaient excellents (acc > 99 %).
En webcam, le CNN 28×28 confondait certaines lettres (p.ex. P/Q/H) à cause du décalage de domaine : main réelle ≠ image 28×28 parfaite (échelle, orientation, lumière, arrière-plan, morphologies de mains différentes).

Pivot : passer à une pipeline temps réel à base de landmarks (repères de la main) :

détection robuste des 21 points MediaPipe Hands,

features invariantes (translation/échelle/rotation/main gauche/droite),

MLP scikit-learn rapide à entraîner,

stabilisation temporelle, tempo lettre→mot→phrase, auto-correction.

Résultat : une démo beaucoup plus stable en conditions réelles, tout en gardant la voie MNIST pour les métriques du rapport.

Fonctionnalités

Détection temps réel d’une main, extraction 21 landmarks.

Classification lettres ASL (24 lettres : A..I, K..Y, sans J/Z dynamiques).

Stabilisation (fenêtre glissante, seuil, cooldown).

Construction mots puis phrases, avec auto-correction (FR/EN).

Fond assombri façon “radiographie” + masque convexe de la main.

Option segmentation de la personne pour neutraliser le décor.

Figures automatiques (matrice de confusion, PR-curves) pour le rapport.

Arborescence du dépôt
AI-SignRecognition/
├─ data/
│  ├─ asl_mnist/                     # (optionnel) Sign Language MNIST
│  ├─ asl_alphabet/                  # ASL Alphabet (1,0 Go décompressé)
│  └─ landmarks/features.csv         # Features extraites (landmarks)
├─ outputs/
│  ├─ models/
│  │  ├─ asl_cnn.pt                  # (optionnel) CNN 28×28
│  │  ├─ asl_class_names.json        # (optionnel) mapping classes CNN
│  │  ├─ landmarks_mlp.joblib        # MLP scikit-learn (landmarks)
│  │  └─ landmark_labels.json        # classes MLP
│  └─ figures/
│     ├─ landmarks_cm.png            # matrice de confusion
│     └─ landmarks_pr.png            # PR-curves
├─ src/
│  ├─ __init__.py
│  ├─ landmark_features.py           # 102 features invariantes
│  ├─ extract_asl_alphabet_landmarks.py
│  ├─ train_landmarks_mlp.py
│  ├─ demo_landmarks.py              # démo lettres simples
│  ├─ demo_words_sentences.py        # démo lettres→mots→phrases (+ XR)
│  ├─ train_asl_multiclass.py        # (optionnel) CNN sur MNIST 28×28
│  ├─ demo_realtime.py               # (optionnel) démo image-based
│  └─ utils_metrics.py               # (optionnel)
├─ main.py                           # (optionnel) pipeline image 28×28
└─ README.md

Pré-requis

Windows 10/11, webcam.

Python géré par uv (environnement virtuel).

Espace disque libre : ≥ 5 Go (le dataset ASL Alphabet est volumineux).

Installation & environnement (uv)

Installer uv (voir site Astral).

Dans le dossier du projet :

uv venv
uv sync


Utiliser Python 3.12 (si besoin) :

uv python install 3.12.11
uv python pin 3.12.11
uv venv --recreate


Dépendances principales (si manquantes) :

uv add opencv-python mediapipe torch torchvision torchaudio
uv add scikit-learn joblib matplotlib seaborn pandas rapidfuzz wordfreq


Note versions : MediaPipe (< 0.11) impose parfois numpy<2. Avec Python 3.12 et uv, la résolution est gérée automatiquement.

Datasets & préparation
Sign Language MNIST (28×28)

Pour le CNN académique (facultatif) :

# Auth Kaggle (une des deux méthodes)
$env:KAGGLE_USERNAME="VOTRE_USERNAME"; $env:KAGGLE_KEY="VOTRE_CLE_API"
# ou bien créer %USERPROFILE%\.kaggle\kaggle.json avec {"username":"...","key":"..."}

# Télécharger
uv run kaggle datasets download -d datamunge/sign-language-mnist -p data\asl_mnist

# Dézipper (méthode Python fiable)
uv run python -c "import zipfile; z=zipfile.ZipFile(r'data\asl_mnist\sign-language-mnist.zip'); z.extractall(r'data\asl_mnist')"

ASL Alphabet (images réalistes) → extraction de landmarks

Dataset plus “réaliste” (arrière-plans variés), idéal pour extraire des landmarks et entraîner le MLP.

# Auth Kaggle (voir ci-dessus), puis :
uv run kaggle datasets download -d grassknoted/asl-alphabet -p data\asl_alphabet --force
uv run python -c "import zipfile; z=zipfile.ZipFile(r'data\asl_alphabet\asl-alphabet.zip'); z.extractall(r'data\asl_alphabet')"
Remove-Item .\data\asl_alphabet\asl-alphabet.zip


Arborescence attendue :

data\asl_alphabet\asl_alphabet_train\asl_alphabet_train\A\*.jpg
...

Voie A — CNN 28×28 (pour le rapport)

Entraînement sur MNIST (24 lettres, sans J/Z) :

uv run python src\train_asl_multiclass.py `
  --data_dir data\asl_mnist `
  --epochs 20 `
  --batch_size 256 `
  --lr 1e-3 `
  --early_patience 5 `
  --use_class_weights true `
  --sampler none


Sorties : outputs/models/asl_cnn.pt, figures (CM, PR-AUC).

Points d’attention :

Ordre des classes doit être strictement le même à l’entraînement et à l’inférence (A..I, K..Y).

J/Z absents (gestes dynamiques).

Très bons scores hors-ligne, mais en caméra le modèle est sensible à l’échelle/rotation/lumière → d’où la Voie B.

Voie B — Landmarks → MLP (pour la démo live)
1) Extraction auto des features

Sans filmer, on parcourt ASL Alphabet, on détecte la main, on calcule 102 features invariantes et on écrit un CSV.

uv run python -m src.extract_asl_alphabet_landmarks `
  --root data\asl_alphabet\asl_alphabet_train\asl_alphabet_train `
  --out_csv data\landmarks\features.csv `
  --max_per_class 400


Fichier créé : data/landmarks/features.csv.

2) Entraînement MLP
uv run python -m src.train_landmarks_mlp


Sorties :

outputs/models/landmarks_mlp.joblib

outputs/models/landmark_labels.json

outputs/figures/landmarks_cm.png, landmarks_pr.png

3) Démo webcam — Lettres → Mots → Phrases

Version simple (lettres stabilisées + mots auto-corrigés + phrase) :

uv run python -m src.demo_words_sentences --lang fr


Règles :

Stabilisation (fenêtre glissante), seuil de confiance, cooldown lettre.

Fin de mot automatique si la main sort du cadre un court instant.

Auto-correction FR/EN via dictionnaire fréquentiel.

Options “radiographie” (fond sombre) et segmentation

Pour focaliser sur la main et atténuer le décor, ajoutez l’assombrissement et le masque convexe.
Exemples :

# Fond sombre + bord doux + hull + squelette
uv run python -m src.demo_words_sentences --lang fr --darken 0.25 --mask_dilate 40 --mask_feather 25

# Idem + segmentation de la personne (fond encore plus propre)
uv run python -m src.demo_words_sentences --lang fr --use_seg --darken 0.25 --mask_dilate 30 --mask_feather 21 --seg_thresh 0.5


Paramètres utiles :

--min_prob 0.85, --stable_frames 12, --letter_cooldown_ms 1200 pour des lettres plus stables.

--word_pause_ms pour la durée sans main qui valide un mot.

--autocorr_thresh pour rendre l’auto-correction plus ou moins stricte.

--lang fr|en selon la langue visée.

Raccourcis clavier & UX

Dans les démos demo_landmarks.py / demo_words_sentences.py :

Q ou Esc : quitter

M : miroir caméra

BKSP : supprimer la dernière lettre en cours

ENTER : valider le mot en cours

SPACE : insérer un espace (et valider le mot en cours)

. , ! ? ; : : ponctuation rapide (valide le mot en cours)

Dépannage (FAQ)

La CLI Kaggle dit “Missing username”
Ajoutez les variables d’environnement dans la session courante :
$env:KAGGLE_USERNAME="..." ; $env:KAGGLE_KEY="..."
ou créez %USERPROFILE%\.kaggle\kaggle.json avec {"username":"...","key":"..."}.

Expand-Archive bloque
Préférez l’extraction Python :
uv run python -c "import zipfile; z=zipfile.ZipFile(r'...zip'); z.extractall(r'...')"

ModuleNotFoundError: landmark_features
a) Ajouter src/__init__.py et lancer avec -m (ex. python -m src.demo_words_sentences)
ou b) exporter PYTHONPATH :
$env:PYTHONPATH=(Resolve-Path .\src).Path

CNN 28×28 : shape mismatch des poids
Vérifiez que le nombre et l’ordre des classes correspondent exactement (24 lettres, sans J/Z).
Erreur typique : poids calculés pour 25 classes vs 24 dans le modèle.

Performance et FPS
Réduire la résolution de capture (par ex. 640×360), limiter --stable_frames, désactiver --use_seg si le CPU est juste.

MediaPipe logs
Les warnings NORM_RECT/XNNPACK sont normaux. Ils n’empêchent pas l’inférence.

Limites connues

Alphabet seulement (lettres statiques A..Y). Les lettres J et Z sont dynamiques et donc exclues ici.

La traduction en phrase est une aide à la saisie (lettres → mot corrigé → phrase), pas une traduction linguistique complète de la LSF/ASL (qui a une grammaire propre).

La robustesse est très bonne en intérieur stable. Éviter les contre-jours extrêmes.

Ce qui a changé en cours de route

Départ sur Sign Language MNIST (28×28) pour la partie DL/TP.

Problèmes en webcam (P/Q/H répétées) → pivot landmarks (plus robuste au réel).

Ajout d’une chaîne de normalisation (main gauche/droite, orientation, échelle) et de features géométriques (102 dims).

Entraînement MLP scikit-learn rapide, puis post-traitement langage (auto-correction, phrase).

Amélioration UX : fond assombri, shape main (convex hull), segmentation optionnelle, stabilisation temporelle.

Licence / crédits

MediaPipe Hands par Google (modèles de détection de mains/landmarks).

Datasets : Sign Language MNIST (Kaggle), ASL Alphabet (Kaggle).

Ce dépôt assemble ces briques pour une démo pédagogique temps réel.

Lancer en 3 commandes (voie B, recommandée)
# 1) Extraire des features à partir d’ASL Alphabet
uv run python -m src.extract_asl_alphabet_landmarks --root data\asl_alphabet\asl_alphabet_train\asl_alphabet_train --out_csv data\landmarks\features.csv --max_per_class 400

# 2) Entraîner le MLP
uv run python -m src.train_landmarks_mlp

# 3) Démo live (lettres → mots → phrase) + fond sombre
uv run python -m src.demo_words_sentences --lang fr --darken 0.25 --mask_dilate 40 --mask_feather 25
