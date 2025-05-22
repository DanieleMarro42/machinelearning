import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="Suggeritore Percorso Universitario", layout="wide")
st.title("🎓 Suggeritore di Percorso Universitario")
st.markdown("""
Questa applicazione interattiva utilizza un algoritmo k-Nearest Neighbors (kNN) per suggerire il corso di laurea più adatto, confrontando i tuoi voti scolastici, interessi personali e preferenze disciplinari con quelli di un dataset di studenti simulati.

🧠 Il modello analizza il tuo profilo confrontandolo con centinaia di altri studenti e ti propone il percorso universitario più adatto in base alla **vicinanza statistica**.
""")

np.random.seed(42)
data = pd.DataFrame({
    'matematica': np.random.randint(1, 10, 1000),
    'italiano': np.random.randint(1, 10, 1000),
    'inglese': np.random.randint(1, 10, 1000),
    'interesse': np.random.choice(['Scientifico', 'Umanistico', 'Artistico'], 1000),
    'preferenza_materia': np.random.choice(['Fisica', 'Storia', 'Arte', 'Biologia', 'Letteratura', 'Tecnologia', 'Lingue'], 1000),
    'corso': np.nan
})


for i, row in data.iterrows():
    if row['interesse'] == 'Scientifico' and row['matematica'] > 7 and row['preferenza_materia'] in ['Fisica', 'Tecnologia', 'Biologia']:
        data.at[i, 'corso'] = 'Ingegneria'
    elif row['interesse'] == 'Umanistico' and row['italiano'] > 7 and row['preferenza_materia'] in ['Storia', 'Letteratura']:
        data.at[i, 'corso'] = 'Lettere'
    elif row['interesse'] == 'Artistico' and row['inglese'] > 6 and row['preferenza_materia'] in ['Arte', 'Lingue']:
        data.at[i, 'corso'] = 'Design'
    else:
        data.at[i, 'corso'] = np.random.choice(['Economia', 'Scienze Politiche', 'Psicologia'])


le_interesse = LabelEncoder()
le_materia = LabelEncoder()
le_corso = LabelEncoder()
data['interesse_num'] = le_interesse.fit_transform(data['interesse'])
data['materia_num'] = le_materia.fit_transform(data['preferenza_materia'])
data['corso_num'] = le_corso.fit_transform(data['corso'])
features = ['matematica', 'italiano', 'inglese', 'interesse_num', 'materia_num']


X = data[features]
y = data['corso']
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

st.sidebar.header("📋 Inserisci i tuoi dati")
matematica = st.sidebar.slider("Voto Matematica", 1, 10, 6)
italiano = st.sidebar.slider("Voto Italiano", 1, 10, 6)
inglese = st.sidebar.slider("Voto Inglese", 1, 10, 6)
interesse = st.sidebar.radio("Area di interesse", ['Scientifico', 'Umanistico', 'Artistico'])


st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Preferenze aggiuntive")
preferenza_materia = st.sidebar.selectbox("Qual è la materia che ti appassiona di più?", ["Fisica", "Storia", "Arte", "Biologia", "Letteratura", "Tecnologia", "Lingue"])

motivazione = st.sidebar.text_area("Perché vuoi scegliere un certo tipo di percorso universitario?", "")


interesse_num = le_interesse.transform([interesse])[0]
materia_num = le_materia.transform([preferenza_materia])[0]
user_input = pd.DataFrame([[matematica, italiano, inglese, interesse_num, materia_num]], columns=features)
prediction = model.predict(user_input)[0]


st.subheader("🎯 Corso Consigliato")
st.success(f"**{prediction}** è il corso consigliato in base al tuo profilo!")

########### 🔁 Iterazioni dettagliate del kNN con esempio esplicito e grafico

from sklearn.metrics.pairwise import euclidean_distances

###
st.subheader("🗂️ Panoramica del dataset degli studenti")

st.markdown("""
In questa sezione esploriamo le caratteristiche principali del dataset utilizzato per l'addestramento del modello kNN. 

Visualizziamo:
- La distribuzione complessiva dei corsi di laurea assegnati
- La relazione tra corsi suggeriti e area di interesse degli studenti

Questi dati forniscono il contesto su cui si basa il funzionamento del modello.
""")

# 📊 Distribuzione dei corsi consigliati
st.subheader("📊 Distribuzione dei corsi consigliati nel dataset")
corsi_count = data['corso'].value_counts()
st.bar_chart(corsi_count)

# 📉 Distribuzione per area di interesse
st.subheader("📉 Distribuzione dei corsi per area di interesse")
fig_dist = px.histogram(data, x="corso", color="interesse", barmode="group",
                        title="Distribuzione dei corsi suggeriti per area di interesse",
                        labels={'interesse': 'Area di Interesse', 'corso': 'Corso suggerito'})
st.plotly_chart(fig_dist, use_container_width=True)

####

st.subheader("🔎 Funzionamento del modello kNN")
st.markdown("""
Questa sezione mostra passo passo come funziona il modello **k-Nearest Neighbors (kNN)** all'interno di questa applicazione.

È stato generato un dataset con 1000 studenti, ognuno con un profilo casuale ma coerente. Ogni studente ha dei voti scolastici, un'area di interesse e una materia preferita. 

Ad esempio:
- Chi ha interesse scientifico, ama Fisica e ha >7 in matematica avrà Ingegneria
- Chi ama l’arte e ha buon voto in inglese avrà Design

Per poter elaborare i dati con l’algoritmo, le variabili testuali (come l’area di interesse o la materia preferita) sono state convertite in formato numerico tramite `LabelEncoder`.

---

### 📊 Le variabili presenti nel dataset

Il dataset contiene 6 variabili principali per ciascuno studente:
- `matematica`
- `italiano`
- `inglese`
- `interesse_num` *(derivato da interesse testuale)*  
- `materia_num` *(derivato da preferenza_materia testuale)* 
- `corso_num` *(etichetta, ovvero il corso di laurea associato)*

---

### 🧠 Le feature usate dal modello (input)

Il modello utilizza le seguenti variabili per calcolare le distanze:
- 🟢 **Voto in Matematica**
- 🟢 **Voto in Italiano**
- 🟢 **Voto in Inglese**
- 🔵 **Interesse** (convertito in valore numerico)
- 🔵 **Materia preferita** (convertita in valore numerico)

La variabile `corso_num` è invece la **variabile target** che il modello cerca di predire.

---

### 📏 Calcolo della distanza

Il modello calcola la **distanza euclidea** tra il tuo profilo e quello di tutti gli altri studenti nel dataset:

$$
\text{distanza} = \sqrt{(x_1 - x'_1)^2 + (x_2 - x'_2)^2 + \dots + (x_n - x'_n)^2}
$$

Più piccola è la distanza, maggiore è la somiglianza.

---

### 👥 Selezione dei vicini

Il modello seleziona i **5 studenti più vicini** al tuo profilo (cioè quelli con distanza minima). 

Questi studenti sono usati per stabilire quale corso consigliare.

---

### 🎓 Scelta del corso suggerito

Il corso di laurea viene scelto in base alla **maggioranza dei corsi** presenti tra i 5 vicini. 

> Esempio: se tra i 5 vicini ci sono 3 studenti che hanno scelto *Psicologia*, il sistema ti suggerirà *Psicologia*.

È come avere un consulente che conosce i percorsi universitari di migliaia di studenti prima di te e, tra quelli con un profilo simile al tuo, prova a intuire quale percorso potrebbe essere il tuo miglior match.
""")


# Calcolo delle distanze
user_vector = user_input.values.reshape(1, -1)
distances = euclidean_distances(X, user_vector).reshape(-1)
data['distanza'] = distances

# Selezione dei 5 più vicini
k = 5
vicini = data.nsmallest(k, 'distanza').copy()

# Visualizzazione tabellare dei vicini
st.markdown("Ecco i 5 studenti più vicini al tuo profilo! (i " + str(k) + " vicini del kNN):")
st.dataframe(vicini[['matematica', 'italiano', 'inglese', 'interesse', 'preferenza_materia', 'corso', 'distanza']].reset_index(drop=True))

# Calcolo PCA se non esiste
if 'PCA1' not in data.columns or 'PCA2' not in data.columns:
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    data['PCA1'] = X_reduced[:, 0]
    data['PCA2'] = X_reduced[:, 1]

# Associa coordinate PCA anche ai vicini
vicini['PCA1'] = data.loc[vicini.index, 'PCA1']
vicini['PCA2'] = data.loc[vicini.index, 'PCA2']

# Calcolo del punto utente
user_pca = pca.transform(user_input)

# Grafico dei vicini su PCA
fig_knn_iter = px.scatter(data, x='PCA1', y='PCA2', color='corso',
                          title=" Rappresentazione grafica - 📍 Posizione dei 5 vicini del tuo profilo (kNN)",
                          opacity=0.3, labels={'corso': 'Corso suggerito'})

# Evidenzia i vicini in rosso
fig_knn_iter.add_scatter(x=vicini['PCA1'], y=vicini['PCA2'],
                         mode='markers+text',
                         marker=dict(size=14, color='red'),
                         text=[f"Vicino {i+1}" for i in range(k)],
                         textposition="top center")

# Evidenzia l'utente in nero
fig_knn_iter.add_scatter(x=[user_pca[0, 0]], y=[user_pca[0, 1]],
                         mode='markers+text',
                         marker=dict(size=16, color='black'),
                         text=['Tu'], textposition='top center')

st.plotly_chart(fig_knn_iter, use_container_width=True)

######

st.markdown("""
Le 5 dimensioni sono ridotte a 2 con PCA (Principal Component Analysis) solo per **mostrare visivamente** la posizione del tuo profilo tra quelli degli altri.
- Ogni punto = uno studente
- Il punto nero = tu
- I vicini visualizzati sono quelli usati per determinare la tua previsione
""")

st.subheader("🧬 Correlazione tra le variabili")
st.markdown("""
La seguente heatmap mostra **quanto ogni variabile è correlata con le altre** nel dataset.

- Una correlazione positiva (vicina a 1) indica che due variabili aumentano insieme.
- Una correlazione negativa (vicina a -1) indica che quando una aumenta, laltra diminuisce.
- Valori vicino a 0 indicano assenza di relazione lineare.

📌 Ad esempio:
- `matematica` e `italiano` sono poco correlate: vuol dire che sapere il voto in matematica **non ti dice molto** sul voto in italiano.
- `inglese` ha una correlazione leggermente negativa con `corso_num`: ciò può indicare che voti alti in inglese **tendono a escludere alcuni corsi** come Ingegneria, e favorirne altri come Design o Lettere.
- `interesse_num` e `corso_num` sono positivamente correlati: significa che **l’area di interesse ha un’influenza diretta sulla scelta del corso**, ed è coerente con la logica di base del dataset simulato.

Questa mappa è utile per capire **quali variabili incidono maggiormente sulla scelta del corso** e **quanto sono indipendenti tra loro**, migliorando così l'efficacia del modello kNN.

Nota: le stesse variabili compaiono sia sull’asse X che sull’asse Y perché la matrice rappresenta **tutte le combinazioni possibili** tra le variabili in input/output. La diagonale mostra sempre `1` perché ogni variabile è perfettamente correlata con sé stessa.
""")

corr = data[['matematica', 'italiano', 'inglese', 'interesse_num', 'materia_num', 'corso_num']].corr()
fig_corr, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig_corr)

st.markdown("---")
st.caption("© 2025 - Progetto didattico sviluppato con Streamlit")