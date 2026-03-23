import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CHARGEMENT ET PRÉPARATION COMMUNE
# ==========================================
print("Chargement des données...")
df = pd.read_csv('movie_metadata.csv')

required_cols = [
    'budget', 'gross', 'title_year', 'duration', 'num_voted_users',
    'content_rating', 'num_critic_for_reviews', 'imdb_score',
    'cast_total_facebook_likes', 'director_facebook_likes',
    'num_user_for_reviews', 'movie_facebook_likes', 'genres',
    'plot_keywords', 'movie_title'
]
df = df.dropna(subset=required_cols)
df = df[df['budget'] > 0]
df = df[df['gross'] > 0]

df['main_genre'] = df['genres'].str.split('|').str[0]
target_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Adventure']
df_clean = df[df['main_genre'].isin(target_genres)].copy()

# Équilibrage par sous-échantillonnage
min_samples = df_clean['main_genre'].value_counts().min()
df_balanced = (
    df_clean
    .groupby('main_genre', group_keys=False)
    .apply(lambda x: x.sample(min_samples, random_state=42))
    .reset_index(drop=True)
)

# Encodage des labels (cohérent entre les deux modèles)
le = LabelEncoder()
le.fit(target_genres)
df_balanced['label'] = le.transform(df_balanced['main_genre'])

# Split partagé : mêmes indices pour NB et BERT → comparaison équitable
indices = np.arange(len(df_balanced))
idx_train, idx_test = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=df_balanced['main_genre']
)

print(f"Équilibrage    : {min_samples} films/genre → {len(df_balanced)} total")
print(f"Train : {len(idx_train)} | Test : {len(idx_test)}\n")

# ==========================================
# 2. NAIVE BAYES — features numériques
# ==========================================
severity_map = {'G': 0, 'PG': 1, 'PG-13': 2, 'R': 8, 'NC-17': 10}
df_balanced['severity_index']     = df_balanced['content_rating'].map(severity_map).fillna(2)
df_balanced['roi']                = ((df_balanced['gross'] - df_balanced['budget']) / (df_balanced['budget'] + 1)).clip(-1, 50)
df_balanced['log_budget']         = np.log1p(df_balanced['budget'])
df_balanced['log_gross']          = np.log1p(df_balanced['gross'])
df_balanced['cost_per_minute']    = df_balanced['budget'] / (df_balanced['duration'] + 1)
df_balanced['movie_age']          = 2026 - df_balanced['title_year']
df_balanced['votes_per_dollar']   = df_balanced['num_voted_users'] / (df_balanced['budget'] / 1000 + 1)
df_balanced['public_vs_critic']   = df_balanced['num_voted_users'] / (df_balanced['num_critic_for_reviews'] + 1)
df_balanced['user_review_ratio']  = df_balanced['num_user_for_reviews'] / (df_balanced['num_voted_users'] + 1)
df_balanced['critic_density']     = df_balanced['num_critic_for_reviews'] / (df_balanced['duration'] + 1)
df_balanced['score_x_votes']      = df_balanced['imdb_score'] * np.log1p(df_balanced['num_voted_users'])
df_balanced['star_power']         = np.log1p(df_balanced['cast_total_facebook_likes'])
df_balanced['director_notoriety'] = np.log1p(df_balanced['director_facebook_likes'])
df_balanced['marketing_intensity']= np.log1p(df_balanced['movie_facebook_likes'])
df_balanced['duration_sq']        = df_balanced['duration'] ** 2
df_balanced['cheap_thrill_score'] = (df_balanced['severity_index'] * df_balanced['roi']) / (df_balanced['budget'] / 1_000_000 + 1)
df_balanced['horror_signal']      = df_balanced['severity_index'] / (df_balanced['budget'] / 1_000_000 + 1)
df_balanced['prestige_score']     = df_balanced['imdb_score'] * df_balanced['duration'] / (df_balanced['budget'] / 1_000_000 + 1)
df_balanced['comedy_roi_signal']  = df_balanced['roi'] / (df_balanced['severity_index'] + 1)
df_balanced['budget_mid_signal']  = 1.0 / (np.abs(df_balanced['log_budget'] - np.log1p(50_000_000)) + 1)
df_balanced['popular_not_prestige']= df_balanced['num_voted_users'] / (df_balanced['imdb_score'] ** 2 + 1)
df_balanced['light_short_score']  = 1.0 / ((df_balanced['duration'] / 90) * (df_balanced['severity_index'] + 1) + 1)

features = [
    'log_budget', 'log_gross', 'roi', 'cost_per_minute', 'movie_age',
    'duration', 'duration_sq', 'severity_index',
    'num_voted_users', 'num_critic_for_reviews', 'num_user_for_reviews',
    'votes_per_dollar', 'public_vs_critic', 'user_review_ratio', 'critic_density',
    'imdb_score', 'score_x_votes', 'star_power', 'director_notoriety', 'marketing_intensity',
    'cheap_thrill_score', 'horror_signal', 'prestige_score',
]

X = df_balanced[features].values
y = df_balanced['main_genre'].values

X_train_nb, X_test_nb = X[idx_train], X[idx_test]
y_train_nb, y_test_nb = y[idx_train], y[idx_test]

pt = PowerTransformer(method='yeo-johnson')
X_train_scaled = pt.fit_transform(X_train_nb)
X_test_scaled  = pt.transform(X_test_nb)

print("=" * 55)
print("MODÈLE 1 — NAIVE BAYES (features numériques)")
print("=" * 55)

param_grid = {'var_smoothing': np.logspace(2, -12, num=300)}
grid = GridSearchCV(
    GaussianNB(), param_grid,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring='accuracy', n_jobs=-1
)
grid.fit(X_train_scaled, y_train_nb)
nb_model  = grid.best_estimator_
nb_preds  = nb_model.predict(X_test_scaled)
nb_acc    = accuracy_score(y_test_nb, nb_preds)

print(f"Meilleur var_smoothing : {grid.best_params_['var_smoothing']:.2e}")
print(f"Accuracy               : {nb_acc:.2%}")
print(classification_report(y_test_nb, nb_preds))

# ==========================================
# 3. BERT — features textuelles (plot_keywords + titre)
# ==========================================
print("=" * 55)
print("MODÈLE 2 — BERT (plot_keywords + titre du film)")
print("=" * 55)

# Texte d'entrée : titre + mots-clés du synopsis (| → espace)
df_balanced['text'] = (
    df_balanced['movie_title'].str.strip() + ' ' +
    df_balanced['plot_keywords'].str.replace('|', ' ', regex=False)
)

texts  = df_balanced['text'].values
labels = df_balanced['label'].values

texts_train, texts_test   = texts[idx_train],  texts[idx_test]
labels_train, labels_test = labels[idx_train], labels[idx_test]


class MovieDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
bert_model = bert_model.to(device)

train_loader = DataLoader(MovieDataset(texts_train, labels_train, tokenizer), batch_size=16, shuffle=True)
test_loader  = DataLoader(MovieDataset(texts_test,  labels_test,  tokenizer), batch_size=16)

EPOCHS      = 4
optimizer   = AdamW(bert_model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

print(f"\nEntraînement BERT ({EPOCHS} epochs)...")
for epoch in range(EPOCHS):
    bert_model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out  = bert_model(
            input_ids=      batch['input_ids'].to(device),
            attention_mask= batch['attention_mask'].to(device),
            labels=         batch['labels'].to(device)
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
    print(f"  Epoch {epoch + 1}/{EPOCHS} — Loss : {total_loss / len(train_loader):.4f}")

# Évaluation BERT
bert_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        out   = bert_model(
            input_ids=      batch['input_ids'].to(device),
            attention_mask= batch['attention_mask'].to(device)
        )
        preds = torch.argmax(out.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch['labels'].numpy())

bert_preds_str = le.inverse_transform(all_preds)
bert_true_str  = le.inverse_transform(all_labels)
bert_acc       = accuracy_score(all_labels, all_preds)

print(f"\nAccuracy : {bert_acc:.2%}")
print(classification_report(bert_true_str, bert_preds_str))

# ==========================================
# 4. COMPARAISON FINALE
# ==========================================
print("\n" + "=" * 55)
print("COMPARAISON FINALE")
print("=" * 55)
print(f"{'Modèle':<35} {'Accuracy':>10}")
print("-" * 47)
print(f"{'Naive Bayes  (features numériques)':<35} {nb_acc:>10.2%}")
print(f"{'BERT         (plot_keywords + titre)':<35} {bert_acc:>10.2%}")
print("-" * 47)

if nb_acc > bert_acc:
    diff = nb_acc - bert_acc
    print(f"\nNaive Bayes gagne de {diff:.2%}")
    print("→ Les signaux numériques (budget, ROI, engagement) sont plus")
    print("  discriminants que le texte court des plot_keywords.")
else:
    diff = bert_acc - nb_acc
    print(f"\nBERT gagne de {diff:.2%}")
    print("→ Le contexte sémantique des mots-clés apporte une information")
    print("  complémentaire que les features numériques ne capturent pas.")
