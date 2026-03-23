import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════
# CONFIG PAGE
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
    padding:1.2rem 1.5rem; border:1px solid #1f2937;
}
.model-title-nb   { color:#4FC3F7; font-size:1.05rem; font-weight:700; margin-bottom:0.8rem; }
.model-title-bert { color:#CE93D8; font-size:1.05rem; font-weight:700; margin-bottom:0.8rem; }
.kw-container { line-height:2.8; padding:0.4rem 0; }
.result-chip {
    display:inline-block; padding:4px 16px; border-radius:20px;
    font-size:1rem; font-weight:700; margin-bottom:0.8rem;
}
.search-result-item {
    padding:0.5rem 1rem; border-radius:8px; cursor:pointer;
    background:#1a1a2e; border:1px solid #2d2d5e;
    margin-bottom:0.3rem; color:#ccc; font-size:0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════
TARGET_GENRES = ['Action', 'Comedy', 'Drama', 'Horror', 'Adventure']
SEVERITY_MAP  = {'G': 0, 'PG': 1, 'PG-13': 2, 'R': 8, 'NC-17': 10}
GENRE_COLORS  = {
    'Action':'#FF6B6B', 'Comedy':'#FFD93D',
    'Drama':'#6BCB77',  'Horror':'#845EC2', 'Adventure':'#4D96FF',
}
FEATURES = [
    'log_budget','log_gross','roi','cost_per_minute','movie_age',
    'duration','duration_sq','severity_index',
    'num_voted_users','num_critic_for_reviews','num_user_for_reviews',
    'votes_per_dollar','public_vs_critic','user_review_ratio','critic_density',
    'imdb_score','score_x_votes','star_power','director_notoriety','marketing_intensity',
    'cheap_thrill_score','horror_signal','prestige_score',
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
    'cheap_thrill_score':'Signal Horror (cheap thrill)',
    'horror_signal':'Signal Horror (rating/budget)','prestige_score':'Score prestige (Drama)',
}


# ══════════════════════════════════════════════════════
# PYTORCH DATASET
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
    df['cheap_thrill_score']  = (df['severity_index']*df['roi'])/(df['budget']/1_000_000+1)
    df['horror_signal']       = df['severity_index']/(df['budget']/1_000_000+1)
    df['prestige_score']      = df['imdb_score']*df['duration']/(df['budget']/1_000_000+1)
    df['text'] = (df['movie_title'].str.strip()+' '+
                  df['plot_keywords'].str.replace('|',' ',regex=False))
    return df


# ══════════════════════════════════════════════════════
# CHARGEMENT SÉPARÉ : NB (rapide) puis BERT (lent)
# ══════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_nb():
    """Charge les données et entraîne uniquement le Naive Bayes (~10s)."""
    df = pd.read_csv('movie_metadata.csv')
    required = ['budget','gross','title_year','duration','num_voted_users','content_rating',
                'num_critic_for_reviews','imdb_score','cast_total_facebook_likes',
                'director_facebook_likes','num_user_for_reviews','movie_facebook_likes',
                'genres','plot_keywords','movie_title']
    df = df.dropna(subset=required)
    df = df[(df['budget']>0)&(df['gross']>0)]
    df['main_genre'] = df['genres'].str.split('|').str[0]
    df = df[df['main_genre'].isin(TARGET_GENRES)].copy()

    min_s  = df['main_genre'].value_counts().min()
    df_bal = (df.groupby('main_genre', group_keys=False)
               .apply(lambda x: x.sample(min_s, random_state=42))
               .reset_index(drop=True))
    df_bal = engineer(df_bal)
    # Nettoyage des espaces insécables (\xa0) dans les titres
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
    grid = GridSearchCV(GaussianNB(), {'var_smoothing': np.logspace(2,-12,num=300)},
                        cv=StratifiedKFold(10,shuffle=True,random_state=42),
                        scoring='accuracy', n_jobs=-1)
    grid.fit(Xtr, y[idx_tr])
    nb     = grid.best_estimator_
    nb_acc = accuracy_score(y[idx_te], nb.predict(Xte))

    return dict(df=df_bal, nb=nb, pt=pt, le=le,
                nb_acc=nb_acc, idx_tr=idx_tr, idx_te=idx_te)


@st.cache_resource(show_spinner=False)
def load_bert(_df_bal, _idx_tr, _idx_te, _le):
    """Entraîne BERT (~5 min sur CPU). Appelé uniquement sur demande."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok    = BertTokenizer.from_pretrained('bert-base-uncased')
    bert   = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5).to(device)

    texts, labels = _df_bal['text'].values, _df_bal['label'].values
    tl = DataLoader(MovieDataset(texts[_idx_tr], labels[_idx_tr], tok), batch_size=16, shuffle=True)
    el = DataLoader(MovieDataset(texts[_idx_te], labels[_idx_te], tok), batch_size=16)

    opt   = AdamW(bert.parameters(), lr=2e-5, weight_decay=0.01)
    steps = len(tl)*4
    sched = get_linear_schedule_with_warmup(opt, steps//10, steps)

    for _ in range(4):
        bert.train()
        for b in tl:
            opt.zero_grad()
            out = bert(input_ids=b['input_ids'].to(device),
                       attention_mask=b['attention_mask'].to(device),
                       labels=b['labels'].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
            opt.step(); sched.step()

    bert.eval()
    ap, al = [], []
    with torch.no_grad():
        for b in el:
            o = bert(b['input_ids'].to(device), b['attention_mask'].to(device))
            ap.extend(torch.argmax(o.logits,1).cpu().numpy())
            al.extend(b['labels'].numpy())
    bert_acc = accuracy_score(al, ap)

    return dict(bert=bert, tokenizer=tok, device=device, bert_acc=bert_acc)


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
    # Utiliser les valeurs scalées (comme le NB les a vues à l'entraînement)
    x  = S['pt'].transform(np.array([row[f] for f in FEATURES]).reshape(1, -1))[0]
    ci = list(nb.classes_).index(pred_genre)
    oi = [i for i, c in enumerate(nb.classes_) if c != pred_genre]
    def log_g(x, mu, var): return -0.5 * ((x - mu)**2 / (var + 1e-9) + np.log(2 * np.pi * (var + 1e-9)))
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
# VISUALISATIONS
# ══════════════════════════════════════════════════════
def proba_chart(proba_dict):
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
        margin=dict(l=0,r=60,t=10,b=0), height=200, font=dict(color='white',size=12),
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
# INTERFACE
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🎬 Genre Predictor</h1>
  <p>Tapez le nom d'un film — les deux modèles prédisent son genre</p>
</div>
""", unsafe_allow_html=True)

# Chargement NB uniquement (rapide)
with st.spinner("⏳ Chargement du Naive Bayes..."):
    NB = load_nb()

# État BERT et film sélectionné dans la session
if 'bert_loaded' not in st.session_state:
    st.session_state.bert_loaded = False
if 'selected_film' not in st.session_state:
    st.session_state.selected_film = ""

st.divider()

# ══════════════════════════════════════════════════════
# ONGLETS : Prédiction | Données | Corrélations
# ══════════════════════════════════════════════════════
tab_pred, tab_data, tab_corr = st.tabs(["🎬 Prédiction", "📊 Données d'entraînement", "🔗 Corrélations"])

with tab_data:
    df_bal = NB['df']
    idx_tr = NB['idx_tr']

    st.markdown("### Distribution des films par genre")
    counts_total = df_bal['main_genre'].value_counts().reset_index()
    counts_total.columns = ['Genre', 'Total dataset']
    train_genres = df_bal.iloc[idx_tr]['main_genre'].value_counts().reset_index()
    train_genres.columns = ['Genre', 'Train']
    test_genres  = df_bal.iloc[NB['idx_te']]['main_genre'].value_counts().reset_index()
    test_genres.columns  = ['Genre', 'Test']

    summary = counts_total.merge(train_genres, on='Genre').merge(test_genres, on='Genre')
    summary['% Train'] = (summary['Train'] / summary['Total dataset'] * 100).round(1).astype(str) + '%'

    # Tableau
    st.dataframe(
        summary.set_index('Genre'),
        use_container_width=True,
    )

    # Bar chart par genre
    fig_dist = go.Figure()
    for col, color in [('Train', '#4FC3F7'), ('Test', '#CE93D8')]:
        fig_dist.add_trace(go.Bar(
            name=col,
            x=summary['Genre'],
            y=summary[col],
            marker_color=color,
            text=summary[col],
            textposition='outside',
        ))
    fig_dist.update_layout(
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), height=320,
        margin=dict(t=20, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("### Statistiques des features (set d'entraînement)")
    train_df = df_bal.iloc[idx_tr][FEATURES + ['main_genre']]
    st.dataframe(
        train_df.groupby('main_genre')[FEATURES].mean().round(3).T
        .style.background_gradient(cmap='RdYlGn', axis=1),
        use_container_width=True,
        height=400,
    )
    st.caption("Moyenne de chaque feature par genre sur le set d'entraînement. Vert = valeur élevée, Rouge = valeur faible.")

with tab_corr:
    st.markdown("### Matrice de corrélation des features")

    corr_labels = [FEATURE_LABELS.get(f, f) for f in FEATURES]
    corr_matrix = NB['df'][FEATURES].corr().values

    fig_corr = go.Figure(go.Heatmap(
        z=corr_matrix,
        x=corr_labels,
        y=corr_labels,
        colorscale='RdBu',
        zmid=0,
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
        texttemplate="%{text}",
        textfont=dict(size=8),
        hoverongaps=False,
    ))
    fig_corr.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=10),
        height=650,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### Top corrélations (valeur absolue > 0.7)")
    corr_df = NB['df'][FEATURES].corr().abs()
    high_corr = []
    for i in range(len(FEATURES)):
        for j in range(i+1, len(FEATURES)):
            val = NB['df'][FEATURES].corr().iloc[i, j]
            if abs(val) > 0.7:
                high_corr.append({
                    'Feature A': FEATURE_LABELS.get(FEATURES[i], FEATURES[i]),
                    'Feature B': FEATURE_LABELS.get(FEATURES[j], FEATURES[j]),
                    'Corrélation': round(val, 3),
                })
    if high_corr:
        hc_df = pd.DataFrame(high_corr).sort_values('Corrélation', key=abs, ascending=False)
        st.dataframe(hc_df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune corrélation > 0.7 trouvée.")

with tab_pred:
    # ── Sélection du film ────────────────────────────────────────────────────
    all_titles  = sorted(NB['df']['movie_title'].str.replace('\xa0', ' ', regex=False).str.strip().unique().tolist())
    options     = [""] + all_titles
    default_idx = 0
    if st.session_state.selected_film in options:
        default_idx = options.index(st.session_state.selected_film)

    selected = st.selectbox("🎬 Choisir un film", options=options, index=default_idx,
                            format_func=lambda x: "— Sélectionner un film —" if x == "" else x)
    if selected:
        st.session_state.selected_film = selected

    if not selected:
        st.markdown("<p style='color:#666;text-align:center;margin-top:2rem;'>Sélectionnez un film pour obtenir une prédiction.</p>", unsafe_allow_html=True)
    else:
        clean_titles   = NB['df']['movie_title'].str.replace('\xa0', ' ', regex=False).str.strip()
        clean_selected = selected.replace('\xa0', ' ').strip()
        matches_row    = NB['df'][clean_titles == clean_selected]

    if matches_row.empty:
        st.warning(f"Film introuvable dans le dataset : **{selected}**")
    else:
        row  = matches_row.iloc[0]
        gcol = GENRE_COLORS.get(row['main_genre'], '#fff')
        real = row['main_genre']

        # ── Fiche film ───────────────────────────────────────────────────
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

        # ── Prédiction NB ────────────────────────────────────────────────
        nb_proba = predict_nb(row, NB)
        nb_pred  = max(nb_proba, key=nb_proba.get)

        col_nb, col_bert = st.columns(2, gap="large")

        with col_nb:
            nc    = GENRE_COLORS.get(nb_pred, '#fff')
            match = "✅" if nb_pred == real else "❌"
            st.markdown(f"""
            <div class="model-box">
              <div class="model-title-nb">🔢 Naive Bayes — Features numériques</div>
              <span class="result-chip" style="background:{nc}22;color:{nc};border:1px solid {nc}55;">
                {nb_pred} &nbsp; {match}
              </span>
              <div style="color:#888;font-size:0.83rem;">
                Confiance : <b style="color:#fff">{nb_proba[nb_pred]:.1%}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Probabilités**")
            st.plotly_chart(proba_chart(nb_proba), use_container_width=True)

            st.markdown(f"**Pourquoi *{nb_pred}* ?**")
            imp = nb_importance(row, nb_pred, NB)
            st.plotly_chart(importance_chart(imp), use_container_width=True)
            st.caption("🔴 Rouge = a favorisé ce genre · 🔵 Bleu = a joué contre")

            with st.expander("Détail des 3 features clés"):
                for fname in imp.index[:3]:
                    val   = row[fname]
                    delta = imp[fname]
                    mu    = NB['nb'].theta_[list(NB['nb'].classes_).index(nb_pred)][FEATURES.index(fname)]
                    st.markdown(
                        f"**{FEATURE_LABELS.get(fname,fname)}** : `{val:.3f}` "
                        f"({'▲ au-dessus' if val>mu else '▼ en dessous'} de la moyenne "
                        f"*{nb_pred}* `{mu:.3f}`) → `{delta:+.3f}`"
                    )

        # ── Colonne BERT ─────────────────────────────────────────────────
        with col_bert:
            st.markdown("""
            <div class="model-box">
              <div class="model-title-bert">🤖 BERT — Plot keywords</div>
            </div>
            """, unsafe_allow_html=True)

            if not st.session_state.bert_loaded:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🚀 Lancer BERT", use_container_width=True, type="primary"):
                    st.session_state.bert_loaded = True
                    st.rerun()
                st.caption("⏳ Entraînement ~5 min sur CPU (une seule fois, mis en cache ensuite)")
            else:
                with st.spinner("Entraînement de BERT en cours... (~5 min)"):
                    BERT = load_bert(NB['df'], NB['idx_tr'], NB['idx_te'], NB['le'])

                bert_proba, toks, attn = predict_bert(row, {**NB, **BERT})
                bert_pred = max(bert_proba, key=bert_proba.get)
                bc    = GENRE_COLORS.get(bert_pred, '#fff')
                match = "✅" if bert_pred == real else "❌"

                st.markdown(f"""
                <span class="result-chip" style="background:{bc}22;color:{bc};border:1px solid {bc}55;">
                  {bert_pred} &nbsp; {match}
                </span>
                <div style="color:#888;font-size:0.83rem;margin-bottom:0.8rem;">
                  Confiance : <b style="color:#fff">{bert_proba[bert_pred]:.1%}</b>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Probabilités**")
                st.plotly_chart(proba_chart(bert_proba), use_container_width=True)

                st.markdown(f"**Pourquoi *{bert_pred}* ? — Mots qui ont influencé BERT**")
                st.markdown(keyword_html(toks, attn), unsafe_allow_html=True)
                st.caption("🔴 Grand et coloré = BERT y a prêté attention · ⚪ Petit et pâle = peu d'influence")

                with st.expander("Top 5 tokens les plus influents"):
                    SKIP  = {'[CLS]','[SEP]','[PAD]'}
                    pairs = sorted(
                        [(t, a) for t, a in zip(toks, attn) if t not in SKIP and not t.startswith('##')],
                        key=lambda x: x[1], reverse=True
                    )
                    mx = pairs[0][1] if pairs else 1
                    for t, sc in pairs[:5]:
                        st.markdown(f"`{t}` &nbsp; {'█' * int(sc/mx*20)} &nbsp; `{sc:.4f}`")

        # ── Bilan ─────────────────────────────────────────────────────────
        st.divider()

        def verdict_card(label, pred, conf, color, border):
            ok = pred == real
            return f"""
            <div style="background:#111827;border-radius:12px;padding:1rem;text-align:center;
                        border-left:4px solid {border};">
              <div style="color:#888;font-size:0.8rem;margin-bottom:0.3rem;">{label}</div>
              <div style="font-size:1.4rem;font-weight:700;color:{color};">{pred}</div>
              {'<div style="color:#6BCB77;font-size:0.85rem;">✅ Correct</div>' if ok
               else f'<div style="color:#FF6B6B;font-size:0.85rem;">❌ (réel : {real})</div>'}
              {'<div style="color:#888;font-size:0.8rem;">Confiance : ' + f'{conf:.1%}</div>' if conf else ''}
            </div>"""

        if st.session_state.bert_loaded:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(verdict_card("Genre réel", real, None, gcol, gcol), unsafe_allow_html=True)
            with c2:
                st.markdown(verdict_card("Naive Bayes", nb_pred, nb_proba[nb_pred],
                                         GENRE_COLORS.get(nb_pred,'#fff'), '#4FC3F7'), unsafe_allow_html=True)
            with c3:
                st.markdown(verdict_card("BERT", bert_pred, bert_proba[bert_pred],
                                         GENRE_COLORS.get(bert_pred,'#fff'), '#CE93D8'), unsafe_allow_html=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(verdict_card("Genre réel", real, None, gcol, gcol), unsafe_allow_html=True)
            with c2:
                st.markdown(verdict_card("Naive Bayes", nb_pred, nb_proba[nb_pred],
                                         GENRE_COLORS.get(nb_pred,'#fff'), '#4FC3F7'), unsafe_allow_html=True)
