import warnings, os
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="Genre Predictor", page_icon="🎬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero { text-align:center; padding:2rem 0 0.5rem; }
.hero h1 {
    font-size:2.4rem; font-weight:700;
    background:linear-gradient(90deg,#FF6B6B,#CE93D8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p { color:#888; font-size:0.95rem; margin-top:0.2rem; }
.movie-card {
    background:linear-gradient(135deg,#1a1a2e,#16213e);
    border:1px solid #2d2d5e; border-radius:16px;
    padding:1.4rem 1.8rem; margin:1rem 0;
}
.movie-title { font-size:1.5rem; font-weight:700; color:#fff; }
.movie-meta  { color:#aaa; font-size:0.88rem; margin-top:0.4rem; }
.badge {
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-size:0.85rem; font-weight:600; margin-top:0.6rem;
}
.model-box {
    background:#111827; border-radius:14px;
    padding:1.2rem 1.5rem; border:1px solid #1f2937; margin-bottom:0.5rem;
}
.model-title-nb   { color:#4FC3F7; font-size:1.05rem; font-weight:700; margin-bottom:0.8rem; }
.model-title-bert { color:#CE93D8; font-size:1.05rem; font-weight:700; margin-bottom:0.8rem; }
.kw-container { line-height:2.8; padding:0.4rem 0; }
.result-chip {
    display:inline-block; padding:4px 16px; border-radius:20px;
    font-size:1rem; font-weight:700; margin-bottom:0.8rem;
}
.verdict-box {
    background:#111827; border-radius:12px; padding:1rem;
    text-align:center; margin-bottom:0.5rem;
}
.feedback-box {
    background:#0f172a; border:1px solid #334155;
    border-radius:14px; padding:1.4rem 1.8rem; margin-top:1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════
TARGET_GENRES  = ['Action/Adventure', 'Comedy', 'Drama', 'Horror']
ACTION_LIKE    = {'Action', 'Adventure'}
SEVERITY_MAP   = {'G': 0, 'PG': 1, 'PG-13': 2, 'R': 8, 'NC-17': 10}
GENRE_COLORS   = {
    'Action/Adventure':'#FF6B6B', 'Comedy':'#FFD93D',
    'Drama':'#6BCB77',            'Horror':'#845EC2',
}
FEATURES = [
    'log_budget','log_gross','roi','cost_per_minute','movie_age',
    'duration','duration_sq','severity_index',
    'num_voted_users','num_critic_for_reviews','num_user_for_reviews',
    'votes_per_dollar','public_vs_critic','user_review_ratio','critic_density',
    'imdb_score','score_x_votes','star_power','director_notoriety','marketing_intensity',
]
FEATURE_LABELS = {
    'log_budget':'Budget','log_gross':'Box-office','roi':'ROI',
    'cost_per_minute':'Coût/minute','movie_age':'Âge du film',
    'duration':'Durée','duration_sq':'Durée²','severity_index':'Sévérité (rating)',
    'num_voted_users':'Votes public','num_critic_for_reviews':'Nb critiques',
    'num_user_for_reviews':'Avis utilisateurs','votes_per_dollar':'Votes/dollar',
    'public_vs_critic':'Public vs Critique','user_review_ratio':'Ratio avis',
    'critic_density':'Densité critiques','imdb_score':'Score IMDB',
    'score_x_votes':'Score×votes','star_power':'Popularité acteurs',
    'director_notoriety':'Notoriété réalisateur','marketing_intensity':'Marketing',
}
FEEDBACK_FILE = 'feedback.csv'

# ══════════════════════════════════════════════════════
# DATASET BERT
# ══════════════════════════════════════════════════════
class MovieDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts, self.labels = list(texts), list(labels)
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids':      enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'labels':         torch.tensor(self.labels[idx], dtype=torch.long)}

# ══════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════
def engineer(df):
    df = df.copy()
    df['severity_index']      = df['content_rating'].map(SEVERITY_MAP).fillna(2)
    df['roi']                 = ((df['gross']-df['budget'])/(df['budget']+1)).clip(-1,50)
    df['log_budget']          = np.log1p(df['budget'])
    df['log_gross']           = np.log1p(df['gross'])
    df['cost_per_minute']     = df['budget']/(df['duration']+1)
    df['movie_age']           = 2026-df['title_year']
    df['votes_per_dollar']    = df['num_voted_users']/(df['budget']/1000+1)
    df['public_vs_critic']    = df['num_voted_users']/(df['num_critic_for_reviews']+1)
    df['user_review_ratio']   = df['num_user_for_reviews']/(df['num_voted_users']+1)
    df['critic_density']      = df['num_critic_for_reviews']/(df['duration']+1)
    df['score_x_votes']       = df['imdb_score']*np.log1p(df['num_voted_users'])
    df['star_power']          = np.log1p(df['cast_total_facebook_likes'])
    df['director_notoriety']  = np.log1p(df['director_facebook_likes'])
    df['marketing_intensity'] = np.log1p(df['movie_facebook_likes'])
    df['duration_sq']         = df['duration']**2
    df['text'] = (df['movie_title'].str.strip()+' '+
                  df['plot_keywords'].str.replace('|',' ',regex=False))
    return df

# ══════════════════════════════════════════════════════
# CHARGEMENT NB (avec intégration du feedback)
# ══════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_nb(retrain_key=0):
    df = pd.read_csv('movie_metadata.csv')
    required = ['budget','gross','title_year','duration','num_voted_users','content_rating',
                'num_critic_for_reviews','imdb_score','cast_total_facebook_likes',
                'director_facebook_likes','num_user_for_reviews','movie_facebook_likes',
                'genres','plot_keywords','movie_title']
    df = df.dropna(subset=required)
    df = df[(df['budget']>0)&(df['gross']>0)]
    def assign_genre(genres_str):
        for g in genres_str.split('|'):
            if g in ACTION_LIKE:
                return 'Action/Adventure'
            if g in TARGET_GENRES:
                return g
        return None

    df['main_genre'] = df['genres'].apply(assign_genre)
    df = df[df['main_genre'].notna()].reset_index(drop=True)

    # Intégration du feedback (réentraînement)
    if retrain_key > 0 and os.path.exists(FEEDBACK_FILE):
        fb = pd.read_csv(FEEDBACK_FILE)
        fb_wrong = fb[fb['correct'].astype(str) == 'False'].dropna(subset=['correct_genre'])
        if not fb_wrong.empty:
            additions = []
            for _, fbrow in fb_wrong.iterrows():
                title = str(fbrow['movie_title']).strip()
                match = df[df['movie_title'].str.strip() == title]
                if not match.empty:
                    new_row = match.iloc[0].copy()
                    new_row['main_genre'] = fbrow['correct_genre']
                    for _ in range(5):          # upweight ×5
                        additions.append(new_row)
            if additions:
                df = pd.concat([df, pd.DataFrame(additions)], ignore_index=True)

    df_bal = (df.groupby('main_genre', group_keys=False)
               .apply(lambda x: x.sample(min(600, len(x)), random_state=42))
               .reset_index(drop=True))
    df_bal = engineer(df_bal)
    df_bal['movie_title'] = df_bal['movie_title'].str.replace('\xa0', ' ', regex=False).str.strip()

    le = LabelEncoder()
    le.fit(TARGET_GENRES)
    df_bal['label'] = le.transform(df_bal['main_genre'])

    idx = np.arange(len(df_bal))
    idx_tr, idx_te = train_test_split(idx, test_size=0.3, random_state=42,
                                      stratify=df_bal['main_genre'])

    X, y = df_bal[FEATURES].values, df_bal['main_genre'].values
    pt   = PowerTransformer(method='yeo-johnson')
    Xtr  = pt.fit_transform(X[idx_tr])
    Xte  = pt.transform(X[idx_te])
    grid = GridSearchCV(GaussianNB(), {'var_smoothing': np.logspace(2,-12,num=200)},
                        cv=StratifiedKFold(5,shuffle=True,random_state=42),
                        scoring='accuracy', n_jobs=-1)
    grid.fit(Xtr, y[idx_tr])
    nb       = grid.best_estimator_
    y_te     = y[idx_te]
    y_pred   = nb.predict(Xte)
    nb_acc   = accuracy_score(y_te, y_pred)

    return dict(df=df_bal, nb=nb, pt=pt, le=le,
                nb_acc=nb_acc, idx_tr=idx_tr, idx_te=idx_te,
                y_te=y_te, y_pred_nb=y_pred)

# ══════════════════════════════════════════════════════
# CHARGEMENT BERT
# ══════════════════════════════════════════════════════
BERT_SAVE_PATH = 'bert_finetuned.pt'
BERT_ACC_PATH  = 'bert_acc.txt'

@st.cache_resource(show_spinner=False)
def load_bert(_df_bal, _idx_tr, _idx_te, _le):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok    = BertTokenizer.from_pretrained('bert-base-uncased')
    bert   = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5, attn_implementation='eager').to(device)

    if os.path.exists(BERT_SAVE_PATH) and os.path.exists(BERT_ACC_PATH):
        bert.load_state_dict(torch.load(BERT_SAVE_PATH, map_location=device))
        bert.eval()
        with open(BERT_ACC_PATH) as f:
            bert_acc = float(f.read().strip())
        al = np.load('bert_al.npy') if os.path.exists('bert_al.npy') else np.array([])
        ap = np.load('bert_ap.npy') if os.path.exists('bert_ap.npy') else np.array([])
        return dict(bert=bert, tokenizer=tok, device=device, bert_acc=bert_acc,
                    y_te_bert=al, y_pred_bert=ap)

    texts, labels = _df_bal['text'].values, _df_bal['label'].values
    tl = DataLoader(MovieDataset(texts[_idx_tr], labels[_idx_tr], tok), batch_size=16, shuffle=True)
    el = DataLoader(MovieDataset(texts[_idx_te], labels[_idx_te], tok), batch_size=16)

    opt        = AdamW(bert.parameters(), lr=2e-5, weight_decay=0.01)
    N_EPOCHS   = 4
    total_steps = len(tl) * N_EPOCHS
    sched      = get_linear_schedule_with_warmup(opt, total_steps//10, total_steps)

    progress_bar = st.progress(0, text="Initialisation de l'entraînement BERT...")
    status_text  = st.empty()
    global_step  = 0

    for epoch in range(N_EPOCHS):
        bert.train()
        epoch_loss = 0.0
        for batch_idx, b in enumerate(tl):
            opt.zero_grad()
            out = bert(input_ids=b['input_ids'].to(device),
                       attention_mask=b['attention_mask'].to(device),
                       labels=b['labels'].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
            opt.step(); sched.step()

            epoch_loss  += out.loss.item()
            global_step += 1
            pct          = global_step / total_steps
            avg_loss     = epoch_loss / (batch_idx + 1)
            progress_bar.progress(pct, text=f"Epoch {epoch+1}/{N_EPOCHS} — batch {batch_idx+1}/{len(tl)} — loss : {avg_loss:.4f}")

        status_text.markdown(
            f"✅ **Epoch {epoch+1}/{N_EPOCHS} terminée** — loss moyenne : `{epoch_loss/len(tl):.4f}`"
        )

    progress_bar.progress(1.0, text="Évaluation en cours...")
    bert.eval()
    ap, al = [], []
    with torch.no_grad():
        for b in el:
            o = bert(b['input_ids'].to(device), b['attention_mask'].to(device))
            ap.extend(torch.argmax(o.logits,1).cpu().numpy())
            al.extend(b['labels'].numpy())
    bert_acc = accuracy_score(al, ap)

    progress_bar.empty()
    status_text.empty()

    torch.save(bert.state_dict(), BERT_SAVE_PATH)
    with open(BERT_ACC_PATH, 'w') as f:
        f.write(str(bert_acc))

    # Sauvegarder les labels et prédictions pour l'évaluation
    np.save('bert_al.npy', np.array(al))
    np.save('bert_ap.npy', np.array(ap))

    return dict(bert=bert, tokenizer=tok, device=device, bert_acc=bert_acc,
                y_te_bert=np.array(al), y_pred_bert=np.array(ap))

# ══════════════════════════════════════════════════════
# PRÉDICTIONS
# ══════════════════════════════════════════════════════
def predict_nb(row, S):
    x   = np.array([row[f] for f in FEATURES]).reshape(1,-1)
    xsc = S['pt'].transform(x)
    p   = S['nb'].predict_proba(xsc)[0]
    return dict(zip(S['nb'].classes_, p))

def nb_importance(row, pred_genre, S):
    nb = S['nb']
    x  = S['pt'].transform(np.array([row[f] for f in FEATURES]).reshape(1, -1))[0]
    ci = list(nb.classes_).index(pred_genre)
    oi = [i for i, c in enumerate(nb.classes_) if c != pred_genre]
    def log_g(x, mu, var): return -0.5*((x-mu)**2/(var+1e-9)+np.log(2*np.pi*(var+1e-9)))
    delta = log_g(x, nb.theta_[ci], nb.var_[ci]) - log_g(x, nb.theta_[oi].mean(0), nb.var_[oi].mean(0))
    return pd.Series(delta, index=FEATURES).sort_values(key=abs, ascending=False)

def predict_bert(row, S):
    enc = S['tokenizer'](row['text'], max_length=64, padding='max_length',
                         truncation=True, return_tensors='pt')
    with torch.no_grad():
        out = S['bert'](input_ids=enc['input_ids'].to(S['device']),
                        attention_mask=enc['attention_mask'].to(S['device']),
                        output_attentions=True)
    prob  = torch.softmax(out.logits,1).squeeze().cpu().numpy()
    attn  = torch.stack([a.squeeze(0) for a in out.attentions[-4:]])
    attn  = attn.mean(dim=(0,1))[0].cpu().numpy()
    tokens = S['tokenizer'].convert_ids_to_tokens(enc['input_ids'].squeeze().tolist())
    return dict(zip(S['le'].classes_, prob)), tokens, attn

# ══════════════════════════════════════════════════════
# FEEDBACK
# ══════════════════════════════════════════════════════
def save_feedback(movie_title, nb_pred, bert_pred, real_genre, correct, correct_genre=None):
    row = {
        'timestamp':     datetime.now().isoformat(),
        'movie_title':   movie_title,
        'nb_pred':       nb_pred,
        'bert_pred':     bert_pred,
        'real_genre':    real_genre,
        'correct':       correct,
        'correct_genre': correct_genre if correct_genre else real_genre,
    }
    new_df = pd.DataFrame([row])
    if os.path.exists(FEEDBACK_FILE):
        new_df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
    else:
        new_df.to_csv(FEEDBACK_FILE, index=False)

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        try:
            return pd.read_csv(FEEDBACK_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=['timestamp','movie_title','nb_pred','bert_pred',
                                  'real_genre','correct','correct_genre'])

# ══════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════
def proba_chart(proba_dict, height=200):
    genres = sorted(proba_dict, key=proba_dict.get, reverse=True)
    vals   = [proba_dict[g]*100 for g in genres]
    fig    = go.Figure(go.Bar(
        x=vals, y=genres, orientation='h',
        marker_color=[GENRE_COLORS[g] for g in genres],
        text=[f"{v:.1f}%" for v in vals], textposition='outside',
        hovertemplate='%{y}: %{x:.1f}%<extra></extra>',
    ))
    fig.update_layout(
        xaxis=dict(range=[0,115], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(autorange='reversed', tickfont=dict(size=13)),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=60,t=10,b=0), height=height, font=dict(color='white',size=12),
    )
    return fig

def importance_chart(imp, n=8):
    top    = imp.head(n)
    labels = [FEATURE_LABELS.get(f,f) for f in top.index]
    vals   = top.values
    fig    = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        marker_color=['#FF6B6B' if v>0 else '#4D96FF' for v in vals],
        text=[f"{v:+.2f}" for v in vals], textposition='outside',
        hovertemplate='%{y}: %{x:+.3f}<extra></extra>',
    ))
    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=True,
                   zerolinecolor='#444', zerolinewidth=1),
        yaxis=dict(autorange='reversed', tickfont=dict(size=11)),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=70,t=10,b=0), height=290, font=dict(color='white',size=11),
    )
    return fig

def keyword_html(tokens, attn):
    SKIP = {'[CLS]','[SEP]','[PAD]'}
    words, scores = [], []
    cw, cs, cc = '', 0.0, 0
    for tok, sc in zip(tokens, attn):
        if tok in SKIP: continue
        if tok.startswith('##'):
            cw += tok[2:]; cs += sc; cc += 1
        else:
            if cw: words.append(cw); scores.append(cs/max(cc,1))
            cw, cs, cc = tok, sc, 1
    if cw: words.append(cw); scores.append(cs/max(cc,1))
    if not scores: return "<i style='color:#666'>Aucun token</i>"
    mx = max(scores) or 1e-9
    parts = []
    for word, sc in zip(words, scores):
        n = sc/mx
        r,g,b = int(255*n), int(80*(1-n)), int(220*(1-n))
        parts.append(
            f'<span style="background:rgba({r},{g},{b},{0.25+0.65*n:.2f});'
            f'color:rgb({min(255,r+60)},{g+40},{b+40});'
            f'padding:4px 9px;border-radius:8px;margin:3px 2px;'
            f'font-size:{0.82+0.55*n:.2f}rem;font-weight:{int(400+300*n)};'
            f'display:inline-block;">{word}</span>'
        )
    return f'<div class="kw-container">{" ".join(parts)}</div>'

# ══════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════
if 'retrain_key' not in st.session_state:
    st.session_state.retrain_key = 0
if 'selected_film' not in st.session_state:
    st.session_state.selected_film = ""

# ══════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🎬 Genre Predictor</h1>
  <p>Sélectionnez un film — Naive Bayes et BERT prédisent son genre simultanément</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# CHARGEMENT NB
# ══════════════════════════════════════════════════════
with st.spinner("⏳ Chargement du modèle Naive Bayes..."):
    NB = load_nb(st.session_state.retrain_key)

st.divider()

# ══════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════
tab_pred, tab_data, tab_corr, tab_eval = st.tabs(["🎬 Prédiction", "📊 Données d'entraînement", "🔗 Corrélations", "📈 Évaluation"])

# ── Onglet Données ──────────────────────────────────────────────────────────
with tab_data:
    df_bal = NB['df']
    idx_tr = NB['idx_tr']

    # ── 1. Distribution train/test ───────────────────────────────────────────
    st.markdown("### 📊 Distribution des films par genre")
    st.caption("Vérifie l'équilibre des classes. Un dataset déséquilibré biaise le modèle vers les genres sur-représentés — ici on vise 600 films par genre pour que Naive Bayes apprenne chaque classe de façon équitable.")
    counts_total  = df_bal['main_genre'].value_counts().reset_index()
    counts_total.columns = ['Genre','Total dataset']
    train_genres  = df_bal.iloc[idx_tr]['main_genre'].value_counts().reset_index()
    train_genres.columns = ['Genre','Train']
    test_genres   = df_bal.iloc[NB['idx_te']]['main_genre'].value_counts().reset_index()
    test_genres.columns  = ['Genre','Test']
    summary = counts_total.merge(train_genres, on='Genre').merge(test_genres, on='Genre')
    summary['% Train'] = (summary['Train']/summary['Total dataset']*100).round(1).astype(str)+'%'
    st.dataframe(summary.set_index('Genre'), use_container_width=True)

    fig_dist = go.Figure()
    for col, color in [('Train','#4FC3F7'),('Test','#CE93D8')]:
        fig_dist.add_trace(go.Bar(name=col, x=summary['Genre'], y=summary[col],
                                  marker_color=color, text=summary[col], textposition='outside'))
    fig_dist.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)',
                           paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                           height=320, margin=dict(t=20,b=0),
                           legend=dict(orientation='h',yanchor='bottom',y=1.02),
                           yaxis=dict(showgrid=False))
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # ── 2. Moyenne des features par genre ────────────────────────────────────
    st.markdown("### 🧮 Moyenne des features par genre")
    st.caption("Montre les profils typiques de chaque genre selon les features numériques. C'est exactement ce que Naive Bayes modélise : les moyennes (μ) et variances (σ²) de chaque feature par classe. Un genre se distinguera bien s'il a des valeurs très différentes des autres.")
    train_df = df_bal.iloc[idx_tr][FEATURES+['main_genre']]
    st.dataframe(
        train_df.groupby('main_genre')[FEATURES].mean().round(3).T
        .style.background_gradient(cmap='RdYlGn', axis=1),
        use_container_width=True, height=400,
    )
    st.caption("Vert = valeur élevée · Rouge = valeur faible. Les colonnes très contrastées sont les features les plus discriminantes.")

    st.divider()

    # ── 3. Boxplots des features clés par genre ──────────────────────────────
    st.markdown("### 📦 Distribution des features clés par genre")
    st.caption("Les boxplots révèlent la dispersion réelle des données, pas seulement la moyenne. Pour Naive Bayes (Gaussien), un bon feature est celui dont les distributions par genre se chevauchent peu — les boîtes bien séparées indiquent un fort pouvoir discriminant.")

    key_features = ['imdb_score', 'log_budget', 'log_gross', 'roi', 'duration', 'severity_index']
    feat_labels  = [FEATURE_LABELS.get(f, f) for f in key_features]
    sel_feat     = st.selectbox("Feature à visualiser", options=key_features,
                                format_func=lambda x: FEATURE_LABELS.get(x, x), key='box_feat')
    fig_box = go.Figure()
    for genre in TARGET_GENRES:
        vals = df_bal[df_bal['main_genre'] == genre][sel_feat].dropna().values
        fig_box.add_trace(go.Box(
            y=vals, name=genre,
            marker_color=GENRE_COLORS.get(genre, '#fff'),
            boxmean='sd', line_width=1.5,
        ))
    fig_box.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), height=400,
        margin=dict(t=20, b=0), yaxis=dict(showgrid=True, gridcolor='#1f2937'),
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # ── 4. Score IMDB moyen par genre ────────────────────────────────────────
    st.markdown("### ⭐ Score IMDB moyen par genre")
    st.caption("Indique si certains genres ont tendance à être mieux notés. Un écart de score entre genres est un signal utile pour le classifieur — si Horror a systématiquement des scores plus bas que Drama, cette feature aide à les distinguer.")
    imdb_means = df_bal.groupby('main_genre')['imdb_score'].agg(['mean','std']).reset_index()
    fig_imdb = go.Figure()
    fig_imdb.add_trace(go.Bar(
        x=imdb_means['main_genre'], y=imdb_means['mean'].round(2),
        error_y=dict(type='data', array=imdb_means['std'].round(2), visible=True),
        marker_color=[GENRE_COLORS.get(g,'#fff') for g in imdb_means['main_genre']],
        text=imdb_means['mean'].round(2), textposition='outside',
    ))
    fig_imdb.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), height=340,
        margin=dict(t=20,b=0), yaxis=dict(range=[0,10], showgrid=True, gridcolor='#1f2937'),
        showlegend=False,
    )
    st.plotly_chart(fig_imdb, use_container_width=True)

    st.divider()

    # ── 5. Budget vs Box-office par genre ────────────────────────────────────
    st.markdown("### 💰 Budget vs Box-office par genre")
    st.caption("Visualise si les genres ont des profils économiques différents. Horror est typiquement le genre avec le meilleur ROI (petit budget, recettes élevées) — ce qui en fait un genre potentiellement bien séparable via les features financières.")
    sample_df = df_bal.sample(min(800, len(df_bal)), random_state=42)
    fig_scatter = go.Figure()
    for genre in TARGET_GENRES:
        sub = sample_df[sample_df['main_genre'] == genre]
        fig_scatter.add_trace(go.Scatter(
            x=sub['log_budget'], y=sub['log_gross'],
            mode='markers', name=genre,
            marker=dict(color=GENRE_COLORS.get(genre,'#fff'), size=5, opacity=0.6),
            hovertemplate=f'<b>{genre}</b><br>Budget log: %{{x:.1f}}<br>Gross log: %{{y:.1f}}<extra></extra>',
        ))
    fig_scatter.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), height=420,
        xaxis=dict(title='Budget (log)', showgrid=True, gridcolor='#1f2937'),
        yaxis=dict(title='Box-office (log)', showgrid=True, gridcolor='#1f2937'),
        margin=dict(t=20,b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ── 6. Années de production par genre ────────────────────────────────────
    st.markdown("### 📅 Années de production par genre")
    st.caption("Montre si certains genres sont plus récents que d'autres. Si Horror se concentre sur des décennies précises, la feature `movie_age` sera discriminante. Un genre très concentré dans le temps aura une distribution narrow — favorable à Naive Bayes Gaussien.")
    fig_year = go.Figure()
    for genre in TARGET_GENRES:
        vals = df_bal[df_bal['main_genre'] == genre]['title_year'].dropna().values
        fig_year.add_trace(go.Histogram(
            x=vals, name=genre,
            marker_color=GENRE_COLORS.get(genre,'#fff'),
            opacity=0.6, nbinsx=30,
        ))
    fig_year.update_layout(
        barmode='overlay',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), height=360,
        margin=dict(t=20,b=0),
        xaxis=dict(title='Année', showgrid=True, gridcolor='#1f2937'),
        yaxis=dict(title='Nombre de films', showgrid=True, gridcolor='#1f2937'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    st.plotly_chart(fig_year, use_container_width=True)

    st.divider()

    # ── 7. Features les plus discriminantes par genre ────────────────────────
    st.markdown("### 🏆 Features qui prédisent le mieux chaque genre")
    st.caption(
        "Pour chaque genre, on calcule la corrélation de Pearson entre chaque feature et "
        "la variable binaire « est-ce ce genre ? » (0/1). Les 3 features avec la plus forte "
        "corrélation absolue sont celles que Naive Bayes exploite le plus pour reconnaître ce genre."
    )

    GENRE_EXPLANATIONS = {
        'Action/Adventure': "Ce groupe se distingue par son **budget élevé** (effets spéciaux, franchises, CGI) et son **box-office fort**. La **popularité des acteurs** (star_power) est un signal fort — ce sont les genres des stars bankables.",
        'Comedy':           "La Comédie a des **budgets modestes** mais un **ROI souvent élevé**. Elle se distingue par une **durée courte** et peu de votes critiques comparé au public.",
        'Drama':            "Le Drame se démarque par son **score IMDB élevé** (films primés) et une **durée longue**. Les critiques y sont plus actifs que le grand public.",
        'Horror':           "L'Horreur est le genre avec le **meilleur ROI** : petit budget, recettes solides. Le **rating de sévérité** (R/NC-17) et un **faible score IMDB** le distinguent nettement.",
    }

    df_enc2 = NB['df'].copy()
    cols_per_row = 3
    genre_chunks = [TARGET_GENRES[i:i+cols_per_row] for i in range(0, len(TARGET_GENRES), cols_per_row)]

    for chunk in genre_chunks:
        cols = st.columns(len(chunk))
        for col, genre in zip(cols, chunk):
            df_enc2['_target'] = (df_enc2['main_genre'] == genre).astype(int)
            corrs = df_enc2[FEATURES + ['_target']].corr()['_target'].drop('_target')
            top3  = corrs.abs().sort_values(ascending=False).head(3).index.tolist()
            gc    = GENRE_COLORS.get(genre, '#fff')
            with col:
                st.markdown(
                    f"<div style='background:#111827;border:1px solid {gc}44;"
                    f"border-left:4px solid {gc};border-radius:12px;padding:1rem;'>"
                    f"<div style='color:{gc};font-weight:700;font-size:1rem;margin-bottom:0.6rem;'>{genre}</div>",
                    unsafe_allow_html=True,
                )
                for rank, feat in enumerate(top3, 1):
                    val = round(corrs[feat], 3)
                    arrow = "▲" if val > 0 else "▼"
                    color = "#6BCB77" if val > 0 else "#FF6B6B"
                    st.markdown(
                        f"<div style='margin-bottom:0.3rem;font-size:0.88rem;'>"
                        f"<span style='color:#888;'>#{rank}</span> "
                        f"<b style='color:#e2e8f0;'>{FEATURE_LABELS.get(feat,feat)}</b> "
                        f"<span style='color:{color};float:right;'>{arrow} {val:+.3f}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption(GENRE_EXPLANATIONS.get(genre, ""))

# ── Onglet Corrélations ──────────────────────────────────────────────────────
with tab_corr:

    # ── 1. Matrice de corrélation ────────────────────────────────────────────
    st.markdown("### 🔥 Matrice de corrélation des features")
    st.caption("Révèle les redondances entre features. Naive Bayes suppose l'indépendance conditionnelle des features — des features très corrélées (|r| > 0.7) violent cette hypothèse et peuvent réduire les performances. Idéalement, on voudrait des features peu corrélées entre elles.")
    corr_labels = [FEATURE_LABELS.get(f,f) for f in FEATURES]
    corr_full   = NB['df'][FEATURES].corr()
    corr_matrix = corr_full.values
    fig_corr = go.Figure(go.Heatmap(
        z=corr_matrix, x=corr_labels, y=corr_labels,
        colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
        texttemplate="%{text}", textfont=dict(size=8), hoverongaps=False,
    ))
    fig_corr.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white',size=10), height=650,
        margin=dict(l=0,r=0,t=20,b=0), xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # ── 2. Top corrélations ──────────────────────────────────────────────────
    st.markdown("### ⚠️ Features fortement corrélées (|r| > 0.7)")
    st.caption("Ces paires violent l'hypothèse d'indépendance de Naive Bayes. Leur présence simultanée dans le modèle peut sur-pondérer certaines dimensions de l'espace de features. À surveiller si les performances sont décevantes.")
    high_corr = []
    for i in range(len(FEATURES)):
        for j in range(i+1, len(FEATURES)):
            val = corr_full.iloc[i,j]
            if abs(val) > 0.7:
                high_corr.append({
                    'Feature A': FEATURE_LABELS.get(FEATURES[i],FEATURES[i]),
                    'Feature B': FEATURE_LABELS.get(FEATURES[j],FEATURES[j]),
                    'Corrélation': round(val,3),
                    'Impact': '🔴 Fort' if abs(val) > 0.9 else '🟠 Modéré',
                })
    if high_corr:
        st.dataframe(pd.DataFrame(high_corr).sort_values('Corrélation',key=abs,ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.success("Aucune corrélation > 0.7 — les features sont suffisamment indépendantes.")

    st.divider()

    # ── 3. Corrélation features vs genre ─────────────────────────────────────
    st.markdown("### 🎯 Pouvoir discriminant des features par genre")
    st.caption("Pour chaque genre, montre quelles features sont les plus corrélées avec l'appartenance à ce genre (encodé 0/1). Une feature avec une corrélation élevée est un bon prédicteur pour ce genre — c'est ce que Naive Bayes exploite via ses distributions conditionnelles.")
    sel_genre = st.selectbox("Genre à analyser", TARGET_GENRES, key='corr_genre')
    df_enc    = NB['df'].copy()
    df_enc['is_genre'] = (df_enc['main_genre'] == sel_genre).astype(int)
    feat_corr = df_enc[FEATURES + ['is_genre']].corr()['is_genre'].drop('is_genre')
    feat_corr = feat_corr.reindex(feat_corr.abs().sort_values(ascending=False).index)
    labels_fc = [FEATURE_LABELS.get(f,f) for f in feat_corr.index]
    fig_fc = go.Figure(go.Bar(
        x=feat_corr.values, y=labels_fc, orientation='h',
        marker_color=['#FF6B6B' if v > 0 else '#4D96FF' for v in feat_corr.values],
        text=[f"{v:+.3f}" for v in feat_corr.values], textposition='outside',
    ))
    fig_fc.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white',size=11), height=500,
        margin=dict(l=0,r=80,t=20,b=0),
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='#444'),
        yaxis=dict(autorange='reversed'),
    )
    st.plotly_chart(fig_fc, use_container_width=True)
    st.caption("🔴 Rouge = corrélation positive (feature élevée → plus probable ce genre) · 🔵 Bleu = corrélation négative")

    st.divider()

    # ── 4. Variance par genre ────────────────────────────────────────────────
    st.markdown("### 📐 Variance des features par genre")
    st.caption("Naive Bayes Gaussien modélise chaque feature par une gaussienne (μ, σ²) par classe. Une faible variance au sein d'un genre signifie que les films de ce genre sont homogènes sur cette feature — la gaussienne sera narrow et bien définie, ce qui améliore la précision du classifieur.")
    var_df = NB['df'].groupby('main_genre')[FEATURES].std().round(3).T
    var_df.index = [FEATURE_LABELS.get(f,f) for f in var_df.index]
    st.dataframe(
        var_df.style.background_gradient(cmap='YlOrRd', axis=1),
        use_container_width=True, height=400,
    )
    st.caption("Jaune = faible variance (gaussienne précise) · Rouge = forte variance (gaussienne large, moins discriminante)")

# ── Onglet Évaluation ────────────────────────────────────────────────────────
with tab_eval:
    y_te      = NB['y_te']
    y_pred_nb = NB['y_pred_nb']

    st.markdown("## 📈 Évaluation des performances")
    st.caption(
        "Les métriques sont calculées sur le **set de test (30%)**, non vu pendant l'entraînement. "
        "Cela garantit une évaluation honnête de la capacité de généralisation des modèles."
    )

    # ── NB Métriques globales ────────────────────────────────────────────────
    st.markdown("### 🔢 Naive Bayes — Métriques globales")
    st.caption(
        "**Accuracy** : part des films correctement classés. "
        "**Precision** : parmi les films prédits comme genre X, combien le sont vraiment. "
        "**Recall** : parmi les vrais films du genre X, combien sont bien détectés. "
        "**F1** : moyenne harmonique de precision et recall — la métrique la plus équilibrée quand les classes ont des tailles différentes."
    )

    nb_report = classification_report(y_te, y_pred_nb, target_names=TARGET_GENRES, output_dict=True)
    metrics_rows = []
    for genre in TARGET_GENRES:
        r = nb_report[genre]
        metrics_rows.append({
            'Genre':     genre,
            'Precision': round(r['precision'], 3),
            'Recall':    round(r['recall'], 3),
            'F1-score':  round(r['f1-score'], 3),
            'Support':   int(r['support']),
        })
    metrics_df = pd.DataFrame(metrics_rows).set_index('Genre')

    col_acc, col_prec, col_rec, col_f1 = st.columns(4)
    col_acc.metric("Accuracy",  f"{nb_report['accuracy']:.1%}")
    col_prec.metric("Precision (moy.)", f"{nb_report['macro avg']['precision']:.1%}")
    col_rec.metric("Recall (moy.)",     f"{nb_report['macro avg']['recall']:.1%}")
    col_f1.metric("F1 (moy.)",          f"{nb_report['macro avg']['f1-score']:.1%}")

    st.dataframe(
        metrics_df.style.background_gradient(cmap='RdYlGn', subset=['Precision','Recall','F1-score']),
        use_container_width=True,
    )

    # ── NB Matrice de confusion ──────────────────────────────────────────────
    st.markdown("#### Matrice de confusion — Naive Bayes")
    st.caption(
        "Chaque cellule (i, j) indique combien de films du genre **réel i** ont été prédits comme genre **j**. "
        "La diagonale = bonnes prédictions. Les cases hors-diagonale révèlent les confusions fréquentes entre genres — "
        "par exemple si Action et Adventure se confondent souvent, c'est que leurs features numériques se ressemblent."
    )
    cm_nb = confusion_matrix(y_te, y_pred_nb, labels=TARGET_GENRES)
    cm_pct = cm_nb.astype(float) / cm_nb.sum(axis=1, keepdims=True)
    fig_cm_nb = go.Figure(go.Heatmap(
        z=cm_pct, x=TARGET_GENRES, y=TARGET_GENRES,
        colorscale='Blues', zmin=0, zmax=1,
        text=[[f"{cm_nb[i][j]}<br>({cm_pct[i][j]:.0%})" for j in range(len(TARGET_GENRES))]
              for i in range(len(TARGET_GENRES))],
        texttemplate="%{text}", textfont=dict(size=11), hoverongaps=False,
    ))
    fig_cm_nb.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=11), height=400,
        margin=dict(l=0,r=0,t=20,b=0),
        xaxis=dict(title='Prédit'), yaxis=dict(title='Réel', autorange='reversed'),
    )
    st.plotly_chart(fig_cm_nb, use_container_width=True)

    # ── NB F1 par genre (radar) ──────────────────────────────────────────────
    st.markdown("#### Radar F1 par genre — Naive Bayes")
    st.caption(
        "Le radar montre d'un coup d'œil les genres bien maîtrisés (pointe large) vs les genres difficiles (pointe courte). "
        "Un genre avec un F1 faible signifie que le modèle confond souvent ce genre avec un autre."
    )
    f1_vals = [nb_report[g]['f1-score'] for g in TARGET_GENRES]
    fig_radar_nb = go.Figure(go.Scatterpolar(
        r=f1_vals + [f1_vals[0]],
        theta=TARGET_GENRES + [TARGET_GENRES[0]],
        fill='toself', line_color='#4FC3F7',
        fillcolor='rgba(79,195,247,0.2)',
        name='F1 NB',
    ))
    fig_radar_nb.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color='#888')),
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=360,
        margin=dict(t=30,b=30),
    )
    st.plotly_chart(fig_radar_nb, use_container_width=True)

    st.divider()

    # ── BERT ─────────────────────────────────────────────────────────────────
    st.markdown("### 🤖 BERT — Métriques globales")
    st.caption("BERT est évalué sur le même set de test que Naive Bayes. Il prédit à partir du **titre + mots-clés** uniquement (texte court, max 64 tokens), ce qui le désavantage par rapport à son usage habituel sur des synopsis longs.")

    bert_ready = os.path.exists(BERT_SAVE_PATH) and os.path.exists('bert_al.npy')
    if not bert_ready:
        st.info("BERT n'a pas encore été entraîné. Va dans l'onglet **🎬 Prédiction**, sélectionne un film — l'entraînement se lancera automatiquement.")
        y_te_bert, y_pred_bert = np.array([]), np.array([])
    else:
        with st.spinner("Chargement BERT pour l'évaluation..."):
            BERT_EVAL = load_bert(NB['df'], NB['idx_tr'], NB['idx_te'], NB['le'])
        y_te_bert   = BERT_EVAL.get('y_te_bert',   np.array([]))
        y_pred_bert = BERT_EVAL.get('y_pred_bert', np.array([]))

    if len(y_te_bert) == 0:
        st.info("Les métriques BERT seront disponibles après le premier entraînement complet. Lance une prédiction pour déclencher l'entraînement.")
    else:
        le = NB['le']
        y_te_bert_names   = le.inverse_transform(y_te_bert.astype(int))
        y_pred_bert_names = le.inverse_transform(y_pred_bert.astype(int))

        bert_report = classification_report(y_te_bert_names, y_pred_bert_names,
                                            target_names=TARGET_GENRES, output_dict=True)
        bert_rows = []
        for genre in TARGET_GENRES:
            r = bert_report[genre]
            bert_rows.append({
                'Genre':     genre,
                'Precision': round(r['precision'], 3),
                'Recall':    round(r['recall'], 3),
                'F1-score':  round(r['f1-score'], 3),
                'Support':   int(r['support']),
            })
        bert_df = pd.DataFrame(bert_rows).set_index('Genre')

        cb_acc, cb_prec, cb_rec, cb_f1 = st.columns(4)
        cb_acc.metric("Accuracy",         f"{bert_report['accuracy']:.1%}")
        cb_prec.metric("Precision (moy.)",f"{bert_report['macro avg']['precision']:.1%}")
        cb_rec.metric("Recall (moy.)",    f"{bert_report['macro avg']['recall']:.1%}")
        cb_f1.metric("F1 (moy.)",         f"{bert_report['macro avg']['f1-score']:.1%}")

        st.dataframe(
            bert_df.style.background_gradient(cmap='RdYlGn', subset=['Precision','Recall','F1-score']),
            use_container_width=True,
        )

        # Matrice de confusion BERT
        st.markdown("#### Matrice de confusion — BERT")
        st.caption("Même lecture que pour NB. Comparer les deux matrices permet de voir si BERT fait les mêmes erreurs que NB ou des erreurs différentes — si les erreurs diffèrent, un ensemble des deux modèles pourrait être plus performant.")
        cm_bert = confusion_matrix(y_te_bert_names, y_pred_bert_names, labels=TARGET_GENRES)
        cm_bert_pct = cm_bert.astype(float) / cm_bert.sum(axis=1, keepdims=True)
        fig_cm_bert = go.Figure(go.Heatmap(
            z=cm_bert_pct, x=TARGET_GENRES, y=TARGET_GENRES,
            colorscale='Purples', zmin=0, zmax=1,
            text=[[f"{cm_bert[i][j]}<br>({cm_bert_pct[i][j]:.0%})" for j in range(len(TARGET_GENRES))]
                  for i in range(len(TARGET_GENRES))],
            texttemplate="%{text}", textfont=dict(size=11), hoverongaps=False,
        ))
        fig_cm_bert.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=11), height=400,
            margin=dict(l=0,r=0,t=20,b=0),
            xaxis=dict(title='Prédit'), yaxis=dict(title='Réel', autorange='reversed'),
        )
        st.plotly_chart(fig_cm_bert, use_container_width=True)

        st.divider()

        # ── Comparaison NB vs BERT ───────────────────────────────────────────
        st.markdown("### ⚖️ Comparaison NB vs BERT par genre")
        st.caption(
            "Permet de voir si BERT et Naive Bayes se complètent. Sur certains genres, NB (features numériques) "
            "peut surpasser BERT (texte court). Sur d'autres, BERT capte des signaux textuels qu'un score numérique ne peut pas voir."
        )
        compare_rows = []
        for genre in TARGET_GENRES:
            compare_rows.append({
                'Genre':    genre,
                'F1 NB':   round(nb_report[genre]['f1-score'], 3),
                'F1 BERT':  round(bert_report[genre]['f1-score'], 3),
                'Meilleur': '🔢 NB' if nb_report[genre]['f1-score'] >= bert_report[genre]['f1-score'] else '🤖 BERT',
            })
        compare_df = pd.DataFrame(compare_rows).set_index('Genre')
        st.dataframe(compare_df.style.background_gradient(cmap='RdYlGn', subset=['F1 NB','F1 BERT']),
                     use_container_width=True)

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name='Naive Bayes', x=TARGET_GENRES,
            y=[nb_report[g]['f1-score'] for g in TARGET_GENRES],
            marker_color='#4FC3F7', text=[f"{nb_report[g]['f1-score']:.2f}" for g in TARGET_GENRES],
            textposition='outside',
        ))
        fig_compare.add_trace(go.Bar(
            name='BERT', x=TARGET_GENRES,
            y=[bert_report[g]['f1-score'] for g in TARGET_GENRES],
            marker_color='#CE93D8', text=[f"{bert_report[g]['f1-score']:.2f}" for g in TARGET_GENRES],
            textposition='outside',
        ))
        fig_compare.update_layout(
            barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), height=380,
            yaxis=dict(range=[0,1.15], showgrid=True, gridcolor='#1f2937', title='F1-score'),
            margin=dict(t=20,b=0),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        st.plotly_chart(fig_compare, use_container_width=True)

# ── Onglet Prédiction ────────────────────────────────────────────────────────
with tab_pred:
    all_titles  = sorted(NB['df']['movie_title'].str.replace('\xa0',' ',regex=False).str.strip().unique().tolist())
    options     = [""]+all_titles
    default_idx = 0
    if st.session_state.selected_film in options:
        default_idx = options.index(st.session_state.selected_film)

    selected = st.selectbox("🎬 Choisir un film", options=options, index=default_idx,
                            format_func=lambda x: "— Sélectionner un film —" if x=="" else x)
    if selected:
        st.session_state.selected_film = selected

    if not selected:
        st.markdown(
            "<p style='color:#666;text-align:center;margin-top:2rem;'>"
            "Sélectionnez un film pour obtenir une prédiction.</p>",
            unsafe_allow_html=True)
        st.stop()

    # Récupérer la ligne du film
    clean_titles   = NB['df']['movie_title'].str.replace('\xa0',' ',regex=False).str.strip()
    clean_selected = selected.replace('\xa0',' ').strip()
    matches_row    = NB['df'][clean_titles == clean_selected]

    if matches_row.empty:
        st.warning(f"Film introuvable dans le dataset : **{selected}**")
        st.stop()

    row  = matches_row.iloc[0]
    gcol = GENRE_COLORS.get(row['main_genre'],'#fff')
    real = row['main_genre']

    # ── Fiche film ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="movie-card">
      <div class="movie-title">{row['movie_title'].strip()}</div>
      <div class="movie-meta">
        {int(row['title_year'])} &nbsp;·&nbsp; {int(row['duration'])} min &nbsp;·&nbsp;
        {row['content_rating']} &nbsp;·&nbsp; ⭐ {row['imdb_score']} IMDB &nbsp;·&nbsp;
        💰 Budget : ${row['budget']/1e6:.0f}M &nbsp;·&nbsp;
        🎟️ Box-office : ${row['gross']/1e6:.0f}M
      </div>
      <div style="margin-top:0.6rem;">
        <span class="badge" style="background:{gcol}22;color:{gcol};border:1px solid {gcol}55;">
          🎭 Genre réel : {real}
        </span>
        <span style="color:#555;font-size:0.82rem;margin-left:1rem;font-style:italic;">
          {row['plot_keywords']}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Prédiction NB ──────────────────────────────────────────────────────
    nb_proba = predict_nb(row, NB)
    nb_pred  = max(nb_proba, key=nb_proba.get)
    imp      = nb_importance(row, nb_pred, NB)

    # ── Colonnes NB | BERT ─────────────────────────────────────────────────
    col_nb, col_bert = st.columns(2, gap="large")

    # — NB —
    with col_nb:
        nc    = GENRE_COLORS.get(nb_pred,'#fff')
        match = "✅" if nb_pred == real else "❌"
        st.markdown(f"""
        <div class="model-box">
          <div class="model-title-nb">🔢 Naive Bayes — Features numériques</div>
          <span class="result-chip" style="background:{nc}22;color:{nc};border:1px solid {nc}55;">
            {nb_pred} &nbsp; {match}
          </span>
          <div style="color:#888;font-size:0.83rem;">
            Confiance : <b style="color:#fff">{nb_proba[nb_pred]:.1%}</b>
            &nbsp;·&nbsp; Précision globale : <b style="color:#fff">{NB['nb_acc']:.1%}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Probabilités par genre**")
        st.plotly_chart(proba_chart(nb_proba), use_container_width=True)

        st.markdown(f"**Features ayant influencé → *{nb_pred}***")
        st.plotly_chart(importance_chart(imp), use_container_width=True)
        st.caption("🔴 Rouge = a favorisé ce genre · 🔵 Bleu = a joué contre")

        with st.expander("Détail des 3 features clés"):
            for fname in imp.index[:3]:
                val = row[fname]
                mu  = NB['nb'].theta_[list(NB['nb'].classes_).index(nb_pred)][FEATURES.index(fname)]
                delta = imp[fname]
                st.markdown(
                    f"**{FEATURE_LABELS.get(fname,fname)}** : `{val:.3f}` "
                    f"({'▲ au-dessus' if val>mu else '▼ en dessous'} de la moyenne "
                    f"*{nb_pred}* `{mu:.3f}`) → `{delta:+.3f}`"
                )

    # — BERT —
    with col_bert:
        st.markdown("""
        <div class="model-box">
          <div class="model-title-bert">🤖 BERT — Analyse des mots-clés</div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🤖 BERT en cours d'initialisation... (~5 min la première fois, mis en cache ensuite)"):
            BERT = load_bert(NB['df'], NB['idx_tr'], NB['idx_te'], NB['le'])

        bert_proba, toks, attn = predict_bert(row, {**NB, **BERT})
        bert_pred = max(bert_proba, key=bert_proba.get)
        bc    = GENRE_COLORS.get(bert_pred,'#fff')
        match = "✅" if bert_pred == real else "❌"

        st.markdown(f"""
        <div class="model-box">
          <span class="result-chip" style="background:{bc}22;color:{bc};border:1px solid {bc}55;">
            {bert_pred} &nbsp; {match}
          </span>
          <div style="color:#888;font-size:0.83rem;">
            Confiance : <b style="color:#fff">{bert_proba[bert_pred]:.1%}</b>
            &nbsp;·&nbsp; Précision globale : <b style="color:#fff">{BERT['bert_acc']:.1%}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Probabilités par genre**")
        st.plotly_chart(proba_chart(bert_proba), use_container_width=True)

        st.markdown(f"**Mots qui ont influencé BERT → *{bert_pred}***")
        st.markdown(keyword_html(toks, attn), unsafe_allow_html=True)
        st.caption("🔴 Grand et coloré = forte influence · ⚪ Petit et pâle = faible influence")

        with st.expander("Top 5 tokens les plus influents"):
            SKIP  = {'[CLS]','[SEP]','[PAD]'}
            pairs = sorted(
                [(t,a) for t,a in zip(toks, attn) if t not in SKIP and not t.startswith('##')],
                key=lambda x: x[1], reverse=True
            )
            mx = pairs[0][1] if pairs else 1
            for t, sc in pairs[:5]:
                st.markdown(f"`{t}` &nbsp; {'█'*int(sc/mx*20)} &nbsp; `{sc:.4f}`")

    # ── Verdict ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🏆 Verdict comparatif")

    def verdict_html(label, pred, conf, color, border, real_genre):
        ok = (pred == real_genre)
        return f"""
        <div class="verdict-box" style="border-left:4px solid {border};">
          <div style="color:#888;font-size:0.8rem;margin-bottom:0.3rem;">{label}</div>
          <div style="font-size:1.4rem;font-weight:700;color:{color};">{pred}</div>
          {'<div style="color:#6BCB77;font-size:0.85rem;">✅ Correct</div>'
           if ok else f'<div style="color:#FF6B6B;font-size:0.85rem;">❌ (réel : {real_genre})</div>'}
          {'<div style="color:#888;font-size:0.8rem;">Confiance : ' + f'{conf:.1%}</div>' if conf else ''}
        </div>"""

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown(verdict_html("🎭 Genre réel", real, None, gcol, gcol, real),
                    unsafe_allow_html=True)
    with v2:
        st.markdown(verdict_html("🔢 Naive Bayes", nb_pred, nb_proba[nb_pred],
                                 GENRE_COLORS.get(nb_pred,'#fff'), '#4FC3F7', real),
                    unsafe_allow_html=True)
    with v3:
        st.markdown(verdict_html("🤖 BERT", bert_pred, bert_proba[bert_pred],
                                 GENRE_COLORS.get(bert_pred,'#fff'), '#CE93D8', real),
                    unsafe_allow_html=True)

