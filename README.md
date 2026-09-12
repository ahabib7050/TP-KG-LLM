# TP — Graphe de connaissances Neo4j & LLM (Graph-RAG)

**Compte rendu** — AIVANCITY, *IA Générative & Empreinte environnementale*

Construction d'un graphe de connaissances Neo4j à partir de données
footballistiques (1872–2026), puis deux pipelines de Graph-RAG permettant à un
LLM de répondre en langage naturel en s'appuyant sur ce graphe comme **source de
vérité** plutôt que sur ses connaissances paramétriques.

Tous les chiffres de ce document sont **mesurés sur l'instance du TP**, pas
estimés. Les écarts constatés avec l'énoncé sont signalés explicitement.

---

## 1. Architecture d'ensemble

```
Question (langage naturel)
        │
        ▼
  [ Traduction en Cypher ]   ← V1 : patrons pré-écrits │ V2 : génération par le LLM
        │
        ▼
  Exécution sur Neo4j (transaction lecture seule, timeout)
        │
        ▼
  Résultats structurés (records)
        │
        ▼
  Injection dans le contexte du 2ᵉ appel LLM
        │
        ▼
  Réponse ancrée dans les données
```

Les deux pipelines sont implémentés comme des **graphes LangGraph** (`StateGraph`),
et non comme des fonctions linéaires. Ce choix apporte trois choses :

- le chemin de refus est une **arête du graphe**, pas un `return` anticipé ;
- la comptabilité des tokens utilise des **réducteurs** (`Annotated[int, operator.add]`),
  chaque nœud ajoutant sa consommation au lieu d'écraser celle du précédent ;
- l'exécution est **traçable** (`stream_mode="updates"`), ce qui rend le pipeline
  inspectable au lieu d'être une boîte noire.

---

## 2. Environnement et reproduction

### Stack

| Composant | Version | Rôle |
|---|---|---|
| Python | 3.12 (via `uv`) | — |
| `neo4j` | 6.3.0 | pilote officiel |
| `pandas` | 3.0.5 | lecture des CSV |
| `langgraph` | 1.2.11 | orchestration des pipelines |
| `anthropic[bedrock]` | — | accès LLM via AWS Bedrock |
| `pytest` | 9.1.1 | tests des garde-fous |

### Installation

```bash
uv sync                     # recrée l'environnement à l'identique depuis uv.lock
```

Configuration : copier `env.example` vers `.env` et renseigner les valeurs.
`.env` est ignoré par git (vérifiable par `git check-ignore -v .env`) :

```
NEO4J_URI="neo4j+s://<instance>.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="..."
NEO4J_DATABASE="<id-instance>"    # voir §6.1
```

Les identifiants AWS proviennent de la chaîne AWS habituelle (profil SSO), pas
du `.env`.

### Lancement

```bash
uv run python database/test-connection.py        # test de connexion
uv run python database/ingest_football_data.py   # ingestion (UNE SEULE FOIS, cf. §3.2)
uv run python database/verify_ingestion.py       # requêtes de contrôle

uv run python -m graphrag.demo                   # V1, routeur LLM
uv run python -m graphrag.demo --regex           # V1, routeur à règles (sans appel API)
uv run python -m graphrag.demo --trace           # V1, trace des nœuds du graphe

uv run python -m graphrag.demo_v2                # V2, génération de Cypher
uv run python -m graphrag.demo_v2 --schema       # schéma introspecté
uv run python -m graphrag.demo_v2 --trace        # V2, trace (montre la boucle de réparation)

uv run python -m pytest tests/ -q                # 22 tests, hors ligne, sans tokens
```

---

## 3. Partie 1 — Construction du graphe

### 3.1 Modélisation

Les CSV sont des lignes plates ; Neo4j attend des entités et des relations.

| Fichier CSV | Devient |
|---|---|
| `results.csv` | un nœud `Match` + liens vers `Team`, `Tournament`, `City`, `Country` |
| `goalscorers.csv` | `Player` + `SCORED_FOR`→Team et `SCORED_IN`→Match |
| `shootouts.csv` | `HAD_SHOOTOUT` Match→équipe gagnante |

**Clé de jointure synthétique.** Les CSV n'ont aucun identifiant de match commun.
Le script en fabrique un : `date_équipeDom_équipeExt`. C'est lui qui permet aux
buts de retrouver leur match (`MATCH (m:Match {id: row.id})`). Noter le `MATCH`
et non `MERGE` : si l'identifiant ne correspond pas, la ligne est **silencieusement
ignorée** — d'où l'importance du contrôle de volumétrie final.

### 3.2 Choix d'ingestion et justification

| Décision | Pourquoi |
|---|---|
| **Index créés avant tout** | Sans index sur les clés de `MERGE`, chaque `MERGE` parcourt tous les nœuds du label. Avec ~50 000 matchs × 5 `MERGE`, le coût devient quadratique. Mesuré : **~2 s par lot de 5 000**, soit ~20 s au total. |
| **Batching par `UNWIND`** | 49 547 requêtes séparées = autant d'allers-retours réseau (~100 ms chacun vers Aura). `UNWIND $batch AS row` déroule la liste **côté serveur** : 10 allers-retours au lieu de 50 000. |
| **`MERGE` pour les entités** | Le Brésil doit être **un seul** nœud référencé par 1 000 matchs — c'est tout l'intérêt d'un graphe. |
| **`CREATE` pour les buts** | Chaque but est un événement distinct portant sa minute, son penalty, son CSC. Un `MERGE` écraserait un triplé en une seule relation. |
| **`FOREACH` comme conditionnelle** | Cypher n'a pas de `IF`. L'idiome `FOREACH(_ IN CASE WHEN … THEN [1] ELSE [] END │ …)` est une boucle sur une liste à 1 ou 0 élément. C'est ainsi que `WON`/`LOST`/`DREW` sont dérivés du score. |

> ⚠️ **Le script n'est pas ré-exécutable.** Les matchs utilisent `MERGE`
> (idempotent) mais les buts utilisent `CREATE`. Une seconde exécution
> dupliquerait les 47 914 relations de buts. Pour repartir de zéro :
> `MATCH (n) DETACH DELETE n`.

### 3.3 Volumétrie obtenue

```
Nœuds     : 67 792                 Relations : 396 006
  Match       49 546                 PART_OF / PLAYED_IN / PLAYED_HOME / PLAYED_AWAY  49 546 ch.
  Player      15 345                 SCORED_FOR / SCORED_IN                           47 914 ch.
  City         2 093                 WON / LOST   38 286 ch.      DREW   22 522
  Team           337                 LOCATED_IN    2 218          HAD_SHOOTOUT  682
  Country        269
  Tournament     202
```

**Écart avec l'énoncé (64 000 nœuds / 340 000 relations) : normal.** L'énoncé
décrit le jeu de données arrêté en 2024 ; l'archive Kaggle téléchargée va jusqu'au
**26/08/2026**, soit ~2 000 matchs supplémentaires.

### 3.4 Schéma

Introspecté **programmatiquement** (`graphrag/schema.py`), jamais codé en dur, en
combinant trois procédures — aucune ne renvoie tout à elle seule :

```
(:City)-[:LOCATED_IN]->(:Country)
(:Match)-[:HAD_SHOOTOUT {first_shooter: STRING, winner: STRING}]->(:Team)
(:Match)-[:PART_OF]->(:Tournament)
(:Match)-[:PLAYED_IN]->(:City)
(:Player)-[:SCORED_FOR {minute: FLOAT, own_goal: BOOLEAN, penalty: BOOLEAN}]->(:Team)
(:Player)-[:SCORED_IN {minute: FLOAT, own_goal: BOOLEAN, penalty: BOOLEAN}]->(:Match)
(:Team)-[:DREW|LOST|PLAYED_AWAY|PLAYED_HOME|WON]->(:Match)
```

### 3.5 Requêtes paramétrées (§1.3)

Implémentées dans `database/verify_ingestion.py` et `graphrag/templates.py`.
**Toujours `$paramètre`, jamais de concaténation** : protection contre l'injection
Cypher, et mise en cache du plan de requête côté Neo4j.

Résultats de contrôle (cohérents avec la réalité, donc graphe correctement câblé) :

```
Meilleurs buteurs — FIFA World Cup : Mbappé 22, Messi 21, Klose 16, Ronaldo 15, Müller 14
Équipes les plus victorieuses       : Brésil 79, Allemagne 70, Argentine 54, France 45, Italie 45
CSC après la 80ᵉ minute             : 136
```

---

## 4. Partie 2.1 — V1 : patrons de requêtes

### Principe

Le LLM **n'écrit jamais de Cypher**. Il fait uniquement de la compréhension du
langage : quelle intention, avec quelles valeurs. Les requêtes sont pré-écrites
et relues à la main dans `graphrag/templates.py`, qui constitue la **frontière de
sécurité** du pipeline.

```
START → route → validate → ⟨exécutable ?⟩ → query → synthesize → END
                                 ↓
                              reject → END
```

Quatre intentions : `top_scorer`, `most_successful_team`, `head_to_head`,
`matches_in_country`.

### Deux appels LLM aux exigences opposées

| | Rôle | Modèle | Pourquoi |
|---|---|---|---|
| Appel 1 | classification + extraction | Haiku 4.5 | tâche contrainte, sortie JSON schématisée, aucune créativité souhaitée |
| Appel 2 | rédaction de la réponse | Opus 4.8 | nécessite de la fluidité |

### L'ancrage est le cœur du dispositif

Le prompt système du 2ᵉ appel impose : *utiliser **uniquement** les données
fournies ; si le résultat est vide, le dire*. Sans cette contrainte, on a construit
une façon coûteuse de laisser le modèle répondre de mémoire — ce que le RAG existe
précisément pour éviter. Sur des questions de football, il aurait souvent raison,
ce qui rend l'échec **difficile à détecter**.

### Trois problèmes que le squelette de l'énoncé masque

1. **Normalisation des entités.** L'utilisateur écrit « la Coupe du Monde » ; le
   graphe contient exactement `"FIFA World Cup"`. Constaté dès le premier appel de
   test : le routeur renvoyait `{"tournament": "Coupe du Monde"}` → **zéro ligne**.
   Correctif : les noms réels (202 tournois, 337 équipes, 269 pays) sont lus dans
   le graphe et injectés dans le prompt du routeur, plus un repli par repliement
   des accents (`fold()`) — le graphe contient `Copa América`, les utilisateurs
   écrivent `Copa America`.
2. **`INTENT_TEMPLATES[intent]` lève `KeyError`.** Le code de l'énoncé suppose une
   intention valide. Une étape `validate_routing()` a été ajoutée entre le LLM et
   la base : intention connue ? paramètres requis présents ? `$limit` borné à 50 ?
3. **`database_="neo4j"` échoue sur cette instance** (voir §6.1).

### Résultats mesurés (5 questions)

| Routeur | Correct | Tokens |
|---|---|---|
| Règles (`--regex`) | **3/5** | 1 802 |
| LLM (Haiku 4.5) | **5/5** | 21 985 |

Les deux échecs du routeur à règles sont instructifs :

- « Qui a marqué le plus de buts en **Coupe du Monde** ? » → `unknown`. Une regex
  ne traduit pas le français vers `FIFA World Cup`. Le routeur LLM y parvient
  grâce à la liste injectée.
- « Which matches did France play in **Brazil** ? » → mal routé vers `head_to_head`,
  car `Brazil` est à la fois une équipe **et** un pays, et la règle teste « deux
  équipes » en premier.

**Le résultat le plus important du TP se trouve dans ce second échec.** Le
synthétiseur avait sous les yeux 5 matchs France–Brésil et a répondu :

> *« the data doesn't specify the venue… So I can't determine which of these
> matches took place in Brazil. »*

Il aurait pu laisser entendre que ces matchs s'étaient joués au Brésil. **La
contrainte d'ancrage a tenu malgré un routage erroné** — c'est la meilleure preuve
empirique que le garde-fou fonctionne.

---

## 5. Partie 2.2 — V2 : génération du Cypher par le LLM

### Principe

Le LLM reçoit le schéma et **écrit lui-même** la requête. La différence avec V1
n'est pas cosmétique : elle déplace le risque. En V1 le modèle choisissait entre
4 requêtes relues ; ici il émet du Cypher arbitraire — d'où l'existence de
`guard.py` et `executor.py`.

```
                  START
                    │
              [generate] ◄─────────────┐   appel LLM 1 : question → Cypher
                    │                  │
                 [guard]               │   analyse statique + LIMIT forcé
                 /     \               │
            [execute]  (rejeté) ───────┤   transaction lecture seule + timeout
             /     \                   │
    [synthesize]  (échec) ─────────────┘   budget de réparation non épuisé ?
          │             \
          │           [give_up]
         END ◄──────────┘
```

**La boucle de réparation** est ce qui rend V2 utilisable : une requête invalide
est renvoyée au modèle **avec l'erreur** pour correction. Elle est **bornée à 3
tentatives** — une boucle non bornée contre une API facturée est un incident de
coût en puissance.

### Garde-fous (§2.2 de l'énoncé) — et lequel tient réellement

| # | Garde-fou exigé | État |
|---|---|---|
| 1 | Compte en lecture seule | ⚠️ **Impossible sur Aura Free** — `SHOW ROLES` et `SHOW USERS` renvoient `Forbidden`. Substitut : transactions `READ_ACCESS`, **vérifié** comme refusant l'écriture côté serveur (`Neo.ClientError.Statement.AccessMode`). |
| 2 | Refus des clauses d'écriture | ✅ `guard.py`, après neutralisation des commentaires et littéraux |
| 3 | `LIMIT` maximale imposée | ✅ ajoutée si absente, ramenée à 50 si supérieure |
| 4 | Timeout + exceptions non propagées | ✅ 10 s côté serveur ; `QueryFailed` ne porte qu'une ligne lisible |

> **L'ordre compte : la couche la plus faible s'exécute en premier.** Une regex sur
> un langage aussi souple que Cypher est en principe contournable. C'est
> `READ_ACCESS`, imposé par Neo4j, qui constitue la garantie réelle ; l'analyse
> textuelle sert à attraper les cas courants tôt et à fournir au modèle une erreur
> exploitable pour se corriger.

Détail d'implémentation : les littéraux de chaîne sont **neutralisés avant** le
scan des mots-clés. Sans cela, une équipe nommée `'DELETE FC'` ferait rejeter une
requête de lecture parfaitement valide.

**22 tests hors ligne** (`tests/test_guard.py`), sans base ni token — les cas
d'attaque comptent davantage que le cas nominal.

### La fuite d'ancrage — découverte et corrigée

Interrogé hors périmètre, le modèle a produit :

```cypher
RETURN 'This question cannot be answered...' AS answer
```

La requête passait la validation, renvoyait **une ligne**, et le synthétiseur la
présentait comme une donnée issue du graphe. **Le contexte « ancré » contenait du
texte écrit par le modèle sur lui-même.** V1 en est structurellement incapable
(ses requêtes sont figées) ; V2 le peut, puisque le modèle rédige la requête.

Correctif — `require_graph_access()` exige désormais un `MATCH` :

```
-> generate
-> guard       query must start with MATCH, OPTIONAL MATCH or CALL {   ← tentative 1 rejetée
-> generate
-> guard
-> execute                                                              ← 0 ligne
-> synthesize  « The query returned no data… »                          ← refus honnête
```

Le refus s'appuie maintenant sur un **résultat vide réel**.

### Résultats mesurés (6 questions)

6/6 correctes, dont **4 hors du périmètre des patrons V1, sans écrire une ligne de
code supplémentaire** :

| Question | Cypher généré (extrait) | Résultat |
|---|---|---|
| Combien de matchs aux tirs au but ? | `MATCH (:Match)-[:HAD_SHOOTOUT]->(:Team) RETURN count(*)` | 682 |
| Quelle ville a accueilli le plus de matchs ? | `…-[:PLAYED_IN]->(c:City) … ORDER BY matches DESC LIMIT 1` | Kuala Lumpur, 748 |
| Combien de CSC après la 80ᵉ ? | `WHERE s.own_goal = true AND s.minute > 80` | 136 |
| Quel joueur a marqué dans le plus de tournois ? | `count(DISTINCT t)` | Cristiano Ronaldo, 6 |

---



## 6. Comparaison V1 / V2 et empreinte

| Critère | V1 — patrons | V2 — schema-aware |
|---|---|---|
| Réponses correctes | 5/5 (dans le périmètre) | 6/6 |
| Tokens (6 questions) | ~26 000 | **2 937** |
| Nouveau type de question | nouveau patron + code | aucun développement |
| Cypher exécuté | 4 requêtes relues | arbitraire, généré |
| Surface d'attaque | choix parmi 4 requêtes | tout Cypher → garde-fous obligatoires |
| Mode d'échec | refuse hors périmètre | requête plausible mais fausse |
| Coût de développement | élevé (1 patron par intention) | faible (1 prompt) |

### ⚠️ V2 est ici ~9× moins cher — ce qui inverse l'intuition

**Cause :** V1 envoie 202 tournois + 337 équipes (~3 900 tokens) à **chaque** appel
du routeur, là où V2 envoie un schéma de ~600 tokens. Ce surcoût est un artefact du
correctif de normalisation (§4), **pas une propriété de l'approche par patrons**.
La conclusion naïve (« les patrons coûtent moins cher ») est donc l'inverse de ce
que montrent les mesures.

### Mise en cache des prompts : mesurée, et inopérante ici

`cache_control` était bien positionné sur le prompt du routeur. Mesure sur trois
appels identiques :

```
haiku  ~3 531 tokens : cache_write=0     cache_read=0        ← aucun cache
haiku  rembourré ~7 129 : cache_write=7129  cache_read=7129   ← cache actif
opus48 ~6 171 tokens : cache_write=6171  cache_read=6171      ← cache actif
```

**Le préfixe minimal cachable de Haiku 4.5 est supérieur à nos 3 531 tokens.** Le
cache ne s'active donc **jamais**, sans aucune erreur — échec silencieux. Les
chiffres V1 ci-dessus sont par conséquent le **pire cas sans cache**.

Leçon transposable : `cache_control` posé ne signifie pas cache actif. Seule la
vérification de `cache_read_input_tokens` le prouve.

### Lecture « empreinte environnementale »

- V1 **peut** router avec un petit modèle (tâche d'extraction) ; V2 **ne le peut
  pas** : écrire du Cypher correct est une tâche de raisonnement. C'est la raison
  de fond pour laquelle V2 est plus coûteux **par token**.
- Mais V2 envoie beaucoup **moins** de tokens. Le bilan net dépend donc entièrement
  de la taille du contexte injecté, pas de l'architecture seule.
- Les profils `eu.` maintiennent l'inférence dans l'UE (résidence des données, et
  mix électrique européen).

---

## 7. Limites et pistes

- **Cache inopérant sur le routeur V1** (§7). Pistes : router sur un modèle au seuil
  plus bas, ou réduire le vocabulaire injecté (top-N tournois + recherche floue).
- **Pas de compte en lecture seule** (Aura Free). Sur une instance payante, créer
  un rôle dédié et l'utiliser pour V2.
- **`head_to_head` ne distingue pas domicile/extérieur** ni n'agrège le bilan global ;
  il renvoie les N derniers matchs.
- **Le nœud incohérent de §6.3** n'est pas corrigé : il est conservé comme
  illustration de qualité des données.
- **Pas d'évaluation systématique** : 5 et 6 questions ne constituent pas un jeu
  d'évaluation. Une comparaison V1/V2 défendable demanderait ~50 questions annotées.

---

## 8. Structure du dépôt

```
data/                          archive Kaggle décompressée (4 CSV)
database/
  test-connection.py           test de connexion
  ingest_football_data.py      ingestion adaptée (CSV locaux + vérification)
  verify_ingestion.py          requêtes paramétrées de contrôle
graphrag/
  config.py                    modèles Bedrock, driver Neo4j, MAX_LIMIT
  templates.py                 4 patrons Cypher + validate_routing()  ← frontière de sécurité V1
  router.py                    Vocabulary, fold(), RegexRouter, LLMRouter
  pipeline.py                  StateGraph V1 : route/validate/query/synthesize/reject
  schema.py                    introspection programmatique du schéma
  guard.py                     validation statique, LIMIT, contrôle d'ancrage
  executor.py                  READ_ACCESS + timeout + assainissement des erreurs
  cypher_v2.py                 StateGraph V2 avec boucle de réparation
  demo.py / demo_v2.py         démonstrations exécutables
tests/test_guard.py            22 tests hors ligne des garde-fous
```

**Convention de code :** toute fonction porte une docstring et des annotations de
type complètes (paramètres **et** retour).
