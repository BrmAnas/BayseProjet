import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import classification_report

# ==========================================
# 1. CHARGEMENT ET NETTOYAGE
# ==========================================
print("Chargement des données...")
df = pd.read_csv('movie_metadata.csv')

# On exploite toutes les colonnes utiles du CSV
required_cols = [
    'budget', 'gross', 'title_year', 'duration', 'num_voted_users',
    'content_rating', 'num_critic_for_reviews', 'imdb_score',
    'cast_total_facebook_likes', 'director_facebook_likes',
    'num_user_for_reviews', 'movie_facebook_likes', 'genres'
]
df = df.dropna(subset=required_cols)
df = df[df['budget'] > 0]
df = df[df['gross'] > 0]

df['main_genre'] = df['genres'].str.split('|').str[0]
target_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Adventure']
df_clean = df[df['main_genre'].isin(target_genres)].copy()

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
severity_map = {'G': 0, 'PG': 1, 'PG-13': 2, 'R': 8, 'NC-17': 10}
df_clean['severity_index'] = df_clean['content_rating'].map(severity_map).fillna(2)

# --- Finances ---
df_clean['roi'] = ((df_clean['gross'] - df_clean['budget']) / (df_clean['budget'] + 1.0)).clip(-1, 50)
df_clean['log_budget']       = np.log1p(df_clean['budget'])
df_clean['log_gross']        = np.log1p(df_clean['gross'])
df_clean['cost_per_minute']  = df_clean['budget'] / (df_clean['duration'] + 1)
df_clean['movie_age']        = 2026 - df_clean['title_year']

# --- Engagement public & critique ---
df_clean['votes_per_dollar']   = df_clean['num_voted_users'] / (df_clean['budget'] / 1000 + 1)
df_clean['public_vs_critic']   = df_clean['num_voted_users'] / (df_clean['num_critic_for_reviews'] + 1)
df_clean['user_review_ratio']  = df_clean['num_user_for_reviews'] / (df_clean['num_voted_users'] + 1)
df_clean['critic_density']     = df_clean['num_critic_for_reviews'] / (df_clean['duration'] + 1)

# --- Qualité & notoriété ---
df_clean['score_x_votes']       = df_clean['imdb_score'] * np.log1p(df_clean['num_voted_users'])
df_clean['star_power']          = np.log1p(df_clean['cast_total_facebook_likes'])
df_clean['director_notoriety']  = np.log1p(df_clean['director_facebook_likes'])
df_clean['marketing_intensity'] = np.log1p(df_clean['movie_facebook_likes'])

# --- Signaux genre-spécifiques ---
df_clean['duration_sq']       = df_clean['duration'] ** 2                                        # pénalise Drama long
df_clean['cheap_thrill_score']= (df_clean['severity_index'] * df_clean['roi']) / (df_clean['budget'] / 1_000_000 + 1)  # Horror
df_clean['horror_signal']     = df_clean['severity_index'] / (df_clean['budget'] / 1_000_000 + 1)                      # Horror cheap R-rated
df_clean['prestige_score']    = df_clean['imdb_score'] * df_clean['duration'] / (df_clean['budget'] / 1_000_000 + 1)   # Drama de prestige

# --- Signaux Comedy ---
# ROI élevé MAIS faible sévérité (≠ Horror qui est R-rated)
df_clean['comedy_roi_signal']    = df_clean['roi'] / (df_clean['severity_index'] + 1)
# Budget modéré : pic autour de 30-80M, s'effondre aux extrêmes (≠ Action gros budget, ≠ Horror micro-budget)
df_clean['budget_mid_signal']    = 1.0 / (np.abs(df_clean['log_budget'] - np.log1p(50_000_000)) + 1)
# Public nombreux mais score IMDB pas exceptionnel (≠ Drama de prestige)
df_clean['popular_not_prestige'] = df_clean['num_voted_users'] / (df_clean['imdb_score'] ** 2 + 1)
# Durée courte + faible sévérité (Comedy rarement > 120 min et rarement NC-17)
df_clean['light_short_score']    = 1.0 / ((df_clean['duration'] / 90) * (df_clean['severity_index'] + 1) + 1)

features = [
    # Finances
    'log_budget', 'log_gross', 'roi', 'cost_per_minute', 'movie_age',
    # Durée
    'duration', 'duration_sq',
    # Rating
    'severity_index',
    # Engagement
    'num_voted_users', 'num_critic_for_reviews', 'num_user_for_reviews',
    'votes_per_dollar', 'public_vs_critic', 'user_review_ratio', 'critic_density',
    # Qualité / popularité
    'imdb_score', 'score_x_votes', 'star_power', 'director_notoriety', 'marketing_intensity',
    # Signaux genre
    'cheap_thrill_score', 'horror_signal', 'prestige_score',
]

# ==========================================
# 3. ÉQUILIBRAGE PAR SOUS-ÉCHANTILLONNAGE
# ==========================================
min_samples = df_clean['main_genre'].value_counts().min()
df_balanced = df_clean.groupby('main_genre', group_keys=False).apply(
    lambda x: x.sample(min_samples, random_state=42), include_groups=False
).reset_index(drop=True)

# On réattache main_genre qui a été exclu par include_groups=False
df_balanced['main_genre'] = df_clean.groupby('main_genre', group_keys=False).apply(
    lambda x: x.sample(min_samples, random_state=42)
)['main_genre'].values

X = df_balanced[features]
y = df_balanced['main_genre']

# ==========================================
# 4. SPLIT → FIT TRANSFORMER → ENTRAÎNEMENT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pt = PowerTransformer(method='yeo-johnson')
X_train_scaled = pt.fit_transform(X_train)   # fit sur train seulement
X_test_scaled  = pt.transform(X_test)        # transform sans refit

# GridSearch avec validation croisée stratifiée et grille fine
param_grid = {'var_smoothing': np.logspace(2, -12, num=300)}
grid = GridSearchCV(
    GaussianNB(), param_grid,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring='accuracy', n_jobs=-1
)
grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_

print(f"\nÉquilibrage    : {min_samples} films par genre ({min_samples * len(target_genres)} total)")
print(f"Meilleur lissage (var_smoothing) : {grid.best_params_['var_smoothing']:.2e}")
print(f"Précision Globale : {best_model.score(X_test_scaled, y_test):.2%}")
print("\nRapport de performance :")
print(classification_report(y_test, best_model.predict(X_test_scaled)))

# ==========================================
# 5. FONCTION DE PRÉDICTION
# ==========================================
def predire_mon_film(titre, budget_millions, box_office_millions, duree_min, annee,
                     age_rating, nb_votes_public, nb_critiques,
                     imdb_score=7.0, cast_fb=5000, director_fb=500,
                     user_reviews=500, movie_fb=10000):

    budget_reel = budget_millions * 1_000_000
    gross_reel  = box_office_millions * 1_000_000
    roi         = np.clip((gross_reel - budget_reel) / (budget_reel + 1), -1, 50)
    severity    = severity_map.get(age_rating, 2)

    vecteur = pd.DataFrame([[
        # Finances
        np.log1p(budget_reel),
        np.log1p(gross_reel),
        roi,
        budget_reel / (duree_min + 1),
        2026 - annee,
        # Durée
        duree_min,
        duree_min ** 2,
        # Rating
        severity,
        # Engagement
        nb_votes_public,
        nb_critiques,
        user_reviews,
        nb_votes_public / (budget_millions * 1000 + 1),
        nb_votes_public / (nb_critiques + 1),
        user_reviews / (nb_votes_public + 1),
        nb_critiques / (duree_min + 1),
        # Qualité / popularité
        imdb_score,
        imdb_score * np.log1p(nb_votes_public),
        np.log1p(cast_fb),
        np.log1p(director_fb),
        np.log1p(movie_fb),
        # Signaux genre
        (severity * roi) / (budget_millions + 1),
        severity / (budget_millions + 1),
        imdb_score * duree_min / (budget_millions + 1),
    ]], columns=features)

    vecteur_scaled = pt.transform(vecteur)
    proba   = best_model.predict_proba(vecteur_scaled)[0]
    classes = best_model.classes_

    resultats = pd.DataFrame({'Genre': classes, 'Probabilité': proba * 100})
    resultats = resultats.sort_values(by='Probabilité', ascending=False)

    print(f"\n==================================================")
    print(f"FILM : {titre.upper()} | {duree_min}min | {age_rating} | IMDB:{imdb_score}")
    print(f"--------------------------------------------------")
    print(resultats.to_string(index=False, formatters={'Probabilité': '{:.1f}%'.format}))




predire_mon_film("The Conjuring",      20,  319, 112, 2013, 'R',    500000,  450, imdb_score=7.5, cast_fb=12000, movie_fb=80000)
predire_mon_film("Fast & Furious 7",  190, 1515, 137, 2015, 'PG-13',380000,  300, imdb_score=7.1, cast_fb=90000, movie_fb=200000)
predire_mon_film("The Hangover",       35,  467, 100, 2009, 'R',    750000,  400, imdb_score=7.7, cast_fb=30000, movie_fb=50000)
predire_mon_film("Dumb and Dumber",    17,  247, 107, 1994, 'PG-13',380000,  150, imdb_score=7.3, cast_fb=20000, movie_fb=10000)
predire_mon_film("Schindler's List",   22,  322, 195, 1993, 'R',   1400000,  200, imdb_score=9.0, cast_fb=15000, movie_fb=30000)
