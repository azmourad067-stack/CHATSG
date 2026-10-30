# Horse Intelligence

Laboratoire d'intelligence prédictive hippique (marché français, courses PMU).
**Usage personnel, outil d'analyse uniquement** : aucun pari automatique, aucune
connexion à un compte opérateur, aucune garantie de gain.

- Audit et architecture : [`docs/00_audit_et_architecture.md`](docs/00_audit_et_architecture.md)
- État d'avancement : **phase 1 (infrastructure de données)**

## Ce qui fonctionne aujourd'hui (phase 1)

| Brique | État |
|---|---|
| Connecteur PMU turfinfo (programme, partants, rapports définitifs) | Implémenté, testé sur des réponses simulées. **Format réel non encore confirmé.** |
| Connecteur Open-Meteo (prévisions/archive) | Implémenté, testé sur des réponses simulées. Non branché aux courses (coordonnées des hippodromes à renseigner). |
| Open PMU API | Code audité ; connecteur limité au **recoupement des arrivées** (une course par jour, cotes non horodatées — voir audit §2.5) |
| Turf.bzh | **Non opérationnel** (payant) |
| HTTP robuste | Délai d'attente, tentatives bornées avec attente exponentielle, débit limité, pas de redirection |
| Base relationnelle | SQLite (local) ou PostgreSQL (`DATABASE_URL`), schéma complet y compris tables des phases suivantes |
| Traçabilité | Chaque réponse brute est conservée (gzip, SHA-256) et référencée par les lignes qui en dérivent |
| Anti-fuite | Partants historisés en ajout seul, requêtes « as-of », cotes relevées après le départ classées `post_start` et exclues, résultats acceptés seulement après le départ |
| Contrôles qualité | Statut par course : vérifiées / partielles / insuffisantes / indisponible |
| Interface Streamlit | Tableau de bord, Courses du jour, Fiche cheval, Configuration. Les pages Analyse, Laboratoire IA et Backtesting affichent explicitement leur indisponibilité. |
| Tests | 58 tests (connecteurs, qualité, stockage, anti-fuite, sécurité, pages Streamlit) + Ruff + Mypy en CI |

Aucune probabilité n'est affichée : aucun modèle n'est encore entraîné ni validé.

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
  de l'historique, à faible cadence (1 requête/s par défaut), avec reprise.
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
réseau requis). Elle confirmera ou corrigera les hypothèses de format et d'unités
listées dans `src/horse_intelligence/data/schemas/pmu.py` et
`src/horse_intelligence/data/processors/normalization.py`.

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
