# Horse Intelligence

Laboratoire d'intelligence prédictive hippique (marché français, courses PMU).
**Usage personnel, outil d'analyse uniquement** : aucun pari automatique, aucune
connexion à un compte opérateur, aucune garantie de gain.

- Audit et architecture : [`docs/00_audit_et_architecture.md`](docs/00_audit_et_architecture.md)
- État d'avancement : phases 1, 2, 3 et **5 (validation)** livrées ; phase 4 (deep learning) différée faute de volume

## Ce qui fonctionne aujourd'hui (phase 1)

| Brique | État |
|---|---|
| Connecteur PMU turfinfo (programme, partants, rapports, masses points de vente et Internet) | **Contrat vérifié contre l'API réelle le 30/10/2026** (voir audit §2.6) ; première journée réelle collectée |
| Connecteur Open-Meteo (prévisions/archive) | Implémenté ; API joignable. Pas encore relié aux courses (coordonnées des hippodromes à renseigner) |
| Open PMU API | Code audité ; connecteur limité au **recoupement des arrivées** (voir audit §2.5). L'hébergeur refuse actuellement l'accès depuis l'environnement de développement (403). |
| Turf.bzh | **Non opérationnel** (payant) |
| HTTP robuste | Délai d'attente, tentatives bornées avec attente exponentielle, débit limité, pas de redirection |
| Base relationnelle | SQLite (local) ou PostgreSQL (`DATABASE_URL`), schéma complet y compris tables des phases suivantes |
| Traçabilité | Chaque réponse brute est conservée (gzip, SHA-256) et référencée par les lignes qui en dérivent |
| Anti-fuite | Partants historisés en ajout seul, requêtes « as-of », cotes relevées après le départ classées `post_start` et exclues, résultats acceptés seulement après le départ |
| Contrôles qualité | Statut par course : vérifiées / partielles / insuffisantes / indisponible |
| Interface Streamlit | Tableau de bord, Courses du jour, Analyse approfondie, Laboratoire IA, Backtesting, Fiche cheval, Configuration |
| Tests | 121 tests (connecteurs, qualité, stockage, anti-fuite, variables, probabilités, walk-forward, incertitude, découverte, simulateur de paris, sécurité, pages Streamlit) + Ruff + Mypy en CI |

## Phase 2 — premier moteur prédictif

| Brique | Contenu |
|---|---|
| Variables « as-of » | Instant de décision = départ − 10 min. Historique interne (cheval, jockey, entraîneur, couple cheval-jockey) limité aux dates **strictement antérieures** ; cotes de référence publiées avant la décision ; musique et cumuls de carrière de la source, dont le caractère « avant course » est contrôlé par un audit automatique à chaque entraînement |
| Probabilités | Calculées par course (softmax), elles somment à 100 % sans renormalisation ad hoc. Places : calcul exact (Harville généralisé) avec le nombre de places payées, qui suit les partants **déclarés** (vérifié sur les rapports réels) |
| Modèles comparés | Uniforme · marché brut · **marché calibré (référence à battre)** · logit conditionnel de forme · LightGBM forme seule · LightGBM combiné au marché |
| Validation | Walk-forward (fenêtre croissante, calibration sur la fin de l'apprentissage, test sur les jours suivants). Mesures : log-loss, Brier, ECE, AUC, top-1/top-3, et bootstrap apparié par course contre le marché calibré |
| Promotion | Un modèle ne devient « actif » que si son gain de log-loss sur le marché calibré a un IC 95 % entièrement négatif sur au moins 300 courses. Sinon, il reste « candidat » : l'interface l'indique et n'affiche aucun signal de valeur |
| Artefacts | JSON + texte LightGBM (jamais pickle), empreinte SHA-256 vérifiée au chargement, version, période et commit enregistrés |

```bash
pip install -r requirements-train.txt
```

```bash
python scripts/train_models.py
```



## Phase 3 — intelligence avancée

| Brique | Contenu |
|---|---|
| Incertitude | Ensemble de modèles appris sur des rééchantillonnages de courses : intervalle par cheval (10e-90e centiles). Score « hors distribution ». Statut de course (exploitable / incertaine / hors distribution) dont les seuils sont appris sur l'historique |
| Explicabilité | Contributions SHAP natives de LightGBM, relatives aux autres partants : forces et faiblesses de chaque cheval (description du modèle, pas une cause) |
| Analyse bayésienne | Jockeys et entraîneurs : victoires observées / attendues par les cotes (A/E), modèle Gamma-Poisson dont le lissage est estimé. Test de dispersion calibré, test de persistance d'une période à l'autre |
| Laboratoire IA | Recherche de sous-groupes (simples et paires) où le marché se trompe : correction des tests multiples (Benjamini-Hochberg) sur tous les tests, puis confirmation sur une période postérieure indépendante. Statuts historisés en base : détectée → en validation → validée / non confirmée / invalidée |
| Anomalies et dérive | Isolation Forest, puis test : les profils atypiques s'écartent-ils des cotes hors échantillon ? Dérive des variables (PSI), en distinguant celles qui évoluent mécaniquement avec l'historique |
| Valeur | Signal affiché seulement si le modèle est validé, la course exploitable, et la borne basse de l'intervalle au-dessus de la probabilité du marché |

```bash
python scripts/run_discovery.py
```

## Phase 5 — validation

| Brique | Contenu |
|---|---|
| Simulateur de paris | Décision sur probabilités hors échantillon et cote de référence antérieure ; règlement au **rapport définitif officiel**. Mise fixe ou Kelly fractionné (théorique). ROI avec IC bootstrap, drawdown, pire série. Courses sans rapport exclues et comptées |
| Anti-surapprentissage | Règle choisie sur une période de sélection, appliquée telle quelle à une période de test postérieure ; repères sans modèle (favori gagnant/placé) |
| Robustesse | Résultats par discipline, hippodrome, distance, taille du champ et période ; sensibilité aux données manquantes et à l'aléa d'apprentissage |
| Audit des fuites | Exécuté sur la base réelle : cotes antérieures à la décision, permutation des résultats futurs, variable trop prédictive, agrégats de la source |

```bash
python scripts/backfill_payouts.py --start 2026-09-20 --end 2026-10-29
```

```bash
python scripts/run_backtest.py
```

Résultats : `docs/03_resultats_phase5.md`. Aucune stratégie n'est rentable hors échantillon.

## Installation locale

```bash
python3.12 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements-dev.txt
```

```bash
python scripts/initialize_database.py
```

```bash
python scripts/refresh_data.py --check-sources
```

```bash
python scripts/refresh_data.py
```

```bash
streamlit run app.py
```

## Collecte des données

- `scripts/refresh_data.py [--date AAAA-MM-JJ]` : collecte une journée. À exécuter
  plusieurs fois les jours de course : chaque passage ajoute un instantané horodaté
  (cotes, non-partants, changements de monte), puis les résultats après les courses.
- `scripts/import_historical_data.py --start AAAA-MM-JJ [--end …]` : constitution
  de l'historique, à faible cadence (1 requête/s par défaut), avec reprise. Options
  `--countries FRA`, `--pools INTERNET` et `--no-payouts` : environ 1 requête par course
  au lieu de 4 (≈ 1 min par journée de courses françaises).
- `scripts/export_parquet.py` : export des tables en Parquet pour les analyses.
- `scripts/crosscheck_open_pmu.py --date AAAA-MM-JJ` : compare les arrivées stockées
  avec l'API communautaire Open PMU (recoupement uniquement).

**Limite importante** : pour une journée passée, les cotes récupérées sont des cotes
d'après-course ; elles ne servent jamais à une décision simulée. Seules les cotes de
référence et celles relevées par nos propres collectes avant le départ le peuvent.

## Configuration

| Variable / secret | Rôle | Défaut |
|---|---|---|
| `DATABASE_URL` | Base de données (`postgresql+psycopg://user:pass@host/db` pour PostgreSQL) | SQLite dans `data_store/` |
| `HI_DATA_DIR` | Dossier des données locales (réponses brutes, SQLite, Parquet) | `data_store/` |

Sources : `config/data_sources.yaml` (activation, débit, délais, tentatives).
Les secrets se placent dans des variables d'environnement ou dans
`.streamlit/secrets.toml` (ignoré par Git), jamais dans le code.

Pour PostgreSQL, installer en plus le pilote : `pip install "psycopg[binary]"`.

## Déploiement Streamlit Community Cloud

1. Pousser le dépôt sur GitHub.
2. Sur share.streamlit.io : *New app* → ce dépôt → fichier principal `app.py`, Python 3.12.
3. Dans *Secrets*, renseigner `DATABASE_URL` (base persistante). Sans cela, l'application
   démarre sur une base SQLite vide et éphémère et l'indique sur le tableau de bord.
4. Déployer, vérifier les journaux, ouvrir le tableau de bord et la page Configuration.

Streamlit Cloud ne sert qu'à la consultation. La collecte planifiée tourne dans
GitHub Actions (`.github/workflows/collect.yml`) dès que le secret de dépôt
`DATABASE_URL` est défini ; les réponses brutes y sont conservées 90 jours comme
artefacts.

## Tests

```bash
pytest -q
```

```bash
pytest -m network
```

La seconde commande lance les tests de contrat contre la vraie API PMU (accès
réseau requis) pour détecter une évolution du format des réponses.

## Structure

```
app.py                         point d'entrée Streamlit
config/data_sources.yaml       registre des sources
src/horse_intelligence/
  config/                      paramètres (env, secrets)
  data/collectors/             connecteurs (base commune, PMU, météo, secondaires)
  data/schemas/                contrats bruts PMU et enregistrements normalisés
  data/processors/             nettoyage, normalisation, contrôles qualité
  data/storage/                schéma SQL, dépôt « as-of », stockage brut, Parquet
  services/                    orchestration de la collecte
  frontend/                    pages Streamlit et composants
  monitoring/                  journalisation avec masquage des secrets
scripts/                       initialisation, collecte, historique, export
tests/                         tests automatisés (fixtures synthétiques)
docs/                          audit et architecture
```

## Jeu responsable

Jouer comporte des risques : endettement, isolement, dépendance.
Joueurs Info Service : 09 74 75 13 13 (appel non surtaxé).
