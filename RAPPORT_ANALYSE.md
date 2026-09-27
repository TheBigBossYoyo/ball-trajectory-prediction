# Rapport d'Analyse Approfondie : Système de Prédiction de Trajectoire

## 1️⃣ Compréhension Globale du Projet

**Objectif du Projet :**
L'objectif principal est de simuler, prédire et visualiser la trajectoire d'une balle rebondissante en utilisant simultanément trois approches distinctes :
1.  **Physique Théorique :** Un modèle déterministe basé sur la Mécanique de Newton.
2.  **Données Expérimentales :** Des mesures réelles collectées via un capteur à ultrasons.
3.  **Intelligence Artificielle :** Un réseau de neurones LSTM (Long Short-Term Memory) entraîné pour prédire les positions futures.

**Problématique :**
Le système aborde la difficulté de modéliser des systèmes dynamiques réels où les idéalisations théoriques (élasticité parfaite, absence de frottements) divergent de la réalité. En juxtaposant un modèle théorique, des données capteurs et une prédiction IA, l'application illustre les forces et limites de chaque méthode pour capturer la physique d'un oscillateur harmonique amorti (une balle perdant de l'énergie).

**Architecture Logicielle :**
Le projet suit une architecture **Client-Serveur** :
*   **Backend (Flask/Python) :** Le "cerveau" calculatoire. Il gère les équations physiques (`physics.py`), traite les données brutes (`data_processing.py`) et exécute le modèle IA (`model.py`).
*   **Frontend (HTML/JS/Three.js) :** La couche de visualisation. Elle génère un environnement 3D et des graphiques 2D pour permettre une comparaison intuitive.

**Pipeline de Données :**
1.  **Acquisition :** Un capteur à ultrasons mesure la distance d'une balle lâchée depuis une hauteur fixe.
2.  **Pré-traitement :** Les données CSV brutes (Temps vs Distance) sont nettoyées (suppression des valeurs aberrantes), inversées (Distance $\to$ Hauteur), lissées (filtre de Savitzky-Golay) et rééchantillonnées sur une grille temporelle uniforme.
3.  **Modélisation/Simulation :**
    *   *Physique :* Intégration numérique des équations du mouvement.
    *   *IA :* Les données traitées alimentent un réseau LSTM pour prédire les pas de temps futurs.
4.  **Visualisation :** L'API envoie les trajectoires au format JSON au frontend pour le rendu 3D et graphique.

---

## 2️⃣ Structure du Code (Arborescence)

### Cœur de l'Application
*   **`app.py`** : Le contrôleur principal. Il initialise le serveur Flask, charge le modèle LSTM et les données capteurs, et expose l'endpoint `/api/simulate`. Il orchestre le flux : réception des paramètres utilisateur $\to$ calcul physique $\to$ traitement capteur $\to$ prédiction IA $\to$ réponse JSON unifiée.
*   **`config.py`** : Configuration centralisée (constantes comme la Gravité $g=9.81$, la Densité de l'air $\rho=1.25$). Assure la cohérence entre les modules.
*   **`schemas.py`** : Définit des contrats de données stricts via **Pydantic**, garantissant la validité des entrées (ex: masse > 0, coefficient de restitution entre 0 et 1).

### Moteurs Physique & Mathématique
*   **`physics.py`** : Le moteur physique. Il définit la classe `Ball` et la fonction `simulate_trajectory_3d` qui exécute la simulation par pas de temps.
*   **`data_processing.py`** : Gère la réalité "bruitée". Contient la logique pour charger les CSV, interpoler les valeurs manquantes (les "glitches" du capteur) et convertir la distance capteur en hauteur balle ($h = H_{capteur} - d$).

### Machine Learning
*   **`model.py`** : Définit l'architecture du Réseau de Neurones. Il construit un LSTM Bidirectionnel avec TensorFlow/Keras, gère la normalisation (mise à l'échelle [0,1]) et la boucle de prédiction.
*   **`synthetic.py`** : Générateur de données. Pour pallier le manque de données réelles, ce script utilise le moteur physique pour générer des milliers de trajectoires "fictives" mais réalistes afin d'entraîner le modèle LSTM.

---

## 3️⃣ Données & Datasets

### Contexte Expérimental
Les fichiers (`data.csv`, `data2.csv`, `data3.csv`) contiennent des mesures temporelles issues d'un capteur à ultrasons monté sur une potence, regardant vers le bas.

**Structure des fichiers :**
*   **`time_after_drop_ms`** : Temps en millisecondes depuis le début de l'acquisition.
*   **`distance_cm`** : La distance mesurée entre le *capteur* et le *sommet de la balle*.

**Transformation Mathématique :**
Pour obtenir la hauteur physique $h(t)$, le code applique :
$$ h(t) = H_{capteur} - d(t) $$
Où $H_{capteur}$ est la hauteur fixe du capteur (configurée à 70.0 cm dans `config.py`).

### Comparaison des Datasets
*   **`data.csv`** : Semble être la référence ("Golden Master"). La trajectoire est propre, montrant un amortissement classique.
*   **`data2.csv` & `data3.csv`** : Représentent d'autres essais, contenant probablement plus de bruit ou des conditions initiales différentes. Le module `data_processing.py` inclut une logique (`clean_sensor_data`) pour gérer les valeurs aberrantes (ex: lecture max de 66.0 cm signifiant une perte de signal), remplacées par interpolation linéaire.

---

## 4️⃣ Modèle Physique

La simulation (`physics.py`) repose sur la **Mécanique Newtonienne Classique**.

### 1. Hypothèses Physiques
*   **Masse Ponctuelle :** La balle est modélisée comme un point de masse $m$. La rotation (effet Magnus) est négligée.
*   **Mouvement Vertical :** Le mouvement principal est selon l'axe Y.
*   **Forces :** Seules la Gravité et la Traînée Aérodynamique sont considérées.

### 2. Bilan des Forces et Équations du Mouvement
La balle est soumise à deux forces :
1.  **Poids ($F_g$) :** Force constante vers le bas.
    $$ \vec{P} = m \vec{g} = (0, -mg, 0) $$
2.  **Traînée ($F_d$) :** Frottement de l'air, opposé au vecteur vitesse $\vec{v}$.
    $$ \vec{F}_d = -\frac{1}{2} \rho C_d A |\vec{v}| \vec{v} $$
    *   $\rho$ : Masse volumique de l'air ($1.25 \, \text{kg/m}^3$).
    *   $C_d$ : Coefficient de traînée (0.47 pour une sphère lisse).
    *   $A$ : Maître-couple (Surface frontale $\pi r^2$).

Application de la Deuxième Loi de Newton (PFD) : $\sum \vec{F} = m \vec{a}$
$$ \vec{a} = \vec{g} - \frac{\rho C_d A}{2m} |\vec{v}| \vec{v} $$

### 3. Intégration Numérique (Velocity Verlet)
Pour résoudre ces équations différentielles, le code utilise l'intégrateur de **Verlet Vitesse**. Cette méthode est préférée à la méthode d'Euler car elle est "symplectique" (elle conserve mieux l'énergie sur le long terme).

$$ \vec{r}(t+\Delta t) = \vec{r}(t) + \vec{v}(t)\Delta t + \frac{1}{2}\vec{a}(t)\Delta t^2 $$
$$ \vec{v}(t+\Delta t) = \vec{v}(t) + \frac{\vec{a}(t) + \vec{a}(t+\Delta t)}{2}\Delta t $$

### 4. Modèle d'Impact et Coefficient de Restitution
Lors du choc avec le sol ($y \le r$), une collision inélastique se produit. La perte d'énergie est régie par le **Coefficient de Restitution ($e$)**.

#### Démonstration de la relation avec la hauteur
Le coefficient $e$ est défini par le rapport des vitesses après et avant impact :
$$ e = \frac{|v_{après}|}{|v_{avant}|} $$

D'après la Conservation de l'Énergie Mécanique (en négligeant les frottements sur la hauteur de chute) :
$$ E_m = E_p + E_c = \text{cste} \implies mgh = \frac{1}{2}mv^2 \implies v = \sqrt{2gh} $$

1.  **Juste avant l'impact $n$ :** La balle tombe de $h_n$.
    $$ v_{descente} = \sqrt{2gh_n} $$
2.  **Juste après l'impact $n$ :** La balle repart avec $v_{remontée}$.
    $$ v_{remontée} = e \cdot v_{descente} = e\sqrt{2gh_n} $$
3.  **Au sommet du rebond $n$ :** La balle atteint $h_{n+1}$.
    $$ v_{remontée} = \sqrt{2gh_{n+1}} $$

En égalant les deux expressions de $v_{remontée}$ :
$$ \sqrt{2gh_{n+1}} = e\sqrt{2gh_n} $$
En élevant au carré :
$$ 2gh_{n+1} = e^2 (2gh_n) \implies h_{n+1} = e^2 h_n $$

D'où l'expression de $e$ :
$$ e = \sqrt{\frac{h_{n+1}}{h_n}} $$

**Pourquoi la racine carrée ?**
Car $e$ est un rapport de **vitesses**, alors que $h$ est une mesure d'**énergie potentielle**. Comme $E_c \propto v^2$ et $E_p \propto h$, le rapport des hauteurs correspond au carré du rapport des vitesses ($e^2$).

#### Suite Géométrique des Hauteurs
La relation de récurrence $h_{n+1} = e^2 h_n$ montre que les hauteurs maximales suivent une suite géométrique de raison $q = e^2$.
*   Rebond $n$ : $h_n = h_0 \cdot (e^2)^n$

Cela explique la décroissance exponentielle observée sur les graphiques. Le code implémente en réalité une restitution décroissante ($e_{eff} = e \cdot 0.95^n$) pour modéliser la fatigue structurelle, ce qui accélère l'amortissement.

---

## 5️⃣ Modèle Machine Learning (LSTM)

Le projet utilise un **Réseau de Neurones Récurrent (RNN)** de type **LSTM (Long Short-Term Memory)**, idéal pour les séries temporelles car il possède une "mémoire" interne.

### Architecture (`model.py`)
1.  **Couche d'Entrée :** Séquence de 50 pas de temps. Features : `[Temps, Hauteur, Vitesse]` (tous normalisés).
2.  **LSTM Bidirectionnel (128 unités) :** Analyse la séquence dans les deux sens (passé $\leftrightarrow$ futur) pour mieux comprendre la dynamique globale.
3.  **Dropout (0.2) :** Désactivation aléatoire de neurones pour éviter le sur-apprentissage (overfitting).
4.  **LSTM (64 unités) :** Seconde couche pour affiner l'extraction de caractéristiques temporelles.
5.  **Couches Dense (Fully Connected) :** Projection finale vers les coordonnées prédites.

### Stratégie d'Entraînement
*   **Données Synthétiques :** `synthetic.py` génère des milliers de trajectoires via le moteur physique en faisant varier aléatoirement $g$, $m$ et $e$. Cela force le LSTM à "apprendre la physique" (les lois générales) plutôt que de mémoriser une seule trajectoire.
*   **Normalisation :** Indispensable. Les entrées sont ramenées dans $[0, 1]$ pour assurer la convergence de la descente de gradient.
*   **Fonction de Coût (Loss) :** MSE (Mean Squared Error), minimise l'écart quadratique moyen entre prédiction et réalité.

---

## 6️⃣ Simulations et Visualisation

### Simulation 3D (Three.js)
Un "Jumeau Numérique" de l'expérience :
*   **Balle Bleue (Physique) :** Suit les équations mathématiques pures.
*   **Balle Verte (Capteur) :** Rejoue les données CSV.
*   **Balle Rouge (IA) :** Affiche la prédiction du réseau de neurones.
Détail réaliste : Les balles s'écrasent verticalement à l'impact (squash & stretch) via une transformation d'échelle inverse.

### Graphique 2D (Chart.js)
Affiche l'évolution $h = f(t)$. On visualise clairement l'**Enveloppe de Décroissance**. Dans un vide parfait, les pics suivraient une exponentielle décroissante $y = A e^{-\lambda t}$. Ce graphique permet de quantifier l'écart entre le modèle idéal (Physique) et la réalité dissipative (Capteur).

---

## 7️⃣ Couche Application

**Stack Technique :**
*   **Backend :** Python, Flask, NumPy, Pandas, TensorFlow.
*   **Frontend :** HTML5, CSS3, JavaScript (ES6+).

**Flux d'Interaction :**
1.  L'utilisateur règle les paramètres (Masse, Gravité, Restitution) sur l'interface web.
2.  Clic sur "Simulate" $\to$ Requête POST `/api/simulate`.
3.  Le backend lance `simulate_trajectory_3d`.
4.  Si activé, le module Sensor rééchantillonne le CSV.
5.  Si activé, le module LSTM effectue l'inférence.
6.  Le frontend reçoit les tableaux JSON et met à jour la scène 3D.

---

## 8️⃣ Bibliothèques Clés

*   **NumPy :** Calcul matriciel et vectoriel haute performance. Utilisé pour les vitesses et forces.
*   **Pandas :** Manipulation de séries temporelles (chargement CSV, interpolation).
*   **TensorFlow/Keras :** Framework de Deep Learning pour le LSTM.
*   **Three.js :** Moteur de rendu 3D WebGL standard du web.
*   **Chart.js :** Bibliothèque de graphiques interactifs.

---

## 9️⃣ Analyse Critique

### Points Forts
1.  **Approche Hybride :** La combinaison Physique / Données / IA offre une vue pédagogique complète.
2.  **Rigueur Physique :** Prise en compte de la traînée quadratique ($F \propto v^2$) et intégration symplectique.
3.  **Gestion des Données :** Le nettoyage des signaux capteurs est robuste.

### Limites
1.  **Fréquence d'Échantillonnage :** Le capteur ultrason a une fréquence limitée (~50Hz), ce qui peut "rater" l'instant exact de l'impact, lissant artificiellement le pic du rebond.
2.  **Coefficient de Traînée Constant :** Le modèle suppose $C_d$ constant. En réalité, il varie avec le nombre de Reynolds.
3.  **Généralisation du LSTM :** Si l'utilisateur choisit des paramètres extrêmes (ex: gravité de Jupiter), le LSTM (entraîné sur des paramètres terrestres) risque de produire des prédictions erronées (hallucinations).

### Améliorations Possibles
*   **Fusion de Capteurs :** Utiliser un Filtre de Kalman pour fusionner les données ultrasons avec un accéléromètre.
*   **Traînée Variable :** Implémenter un $C_d(v)$ dynamique.
*   **Apprentissage en Ligne :** Permettre au LSTM de s'adapter en temps réel aux nouvelles données capteurs reçues.
