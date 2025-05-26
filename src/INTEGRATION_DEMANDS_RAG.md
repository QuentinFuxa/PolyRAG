# Intégration des Demandes dans le Système RAG

## Vue d'ensemble

Cette intégration permet au système RAG de détecter et classifier automatiquement les sections et demandes dans les lettres de suite d'inspection, fusionnant les capacités de `sections_demands.py` directement dans le processus d'indexation RAG.

## Modifications apportées

### 1. Schéma de base de données

**Fichier**: `src/migrations/add_content_classification_to_rag.sql`

Ajout de 3 nouvelles colonnes à la table `rag_document_blocks`:
- `content_type` : Type de contenu ('section_header', 'demand', 'regular')
- `section_type` : Type de section si applicable ('synthesis', 'demands', 'information', etc.)
- `demand_priority` : Priorité de la demande (1 pour prioritaires, 2 pour complémentaires)

### 2. Module de classification

**Fichier**: `src/content_classifiers.py`

- `ContentClassifier` : Classe qui encapsule la logique de détection des sections et demandes
- Utilise les mêmes regex que `sections_demands.py`
- Méthodes principales :
  - `is_letter_de_suite()` : Détecte si un document est une lettre de suite
  - `classify_block()` : Classifie un bloc (section header, demande, ou contenu régulier)

### 3. Intégration dans RAGSystem

**Fichier**: `src/rag_system.py`

Modifications principales:

#### Dans `index_document()`:
1. Détection automatique des lettres de suite
2. Classification des blocs si c'est une lettre de suite
3. Stockage des métadonnées de classification lors de l'insertion

#### Dans les méthodes de recherche:
1. Les méthodes `query()`, `search()`, `_text_search()`, `_embedding_search()`, et `_hybrid_search()` acceptent maintenant les paramètres de filtrage
2. Les filtres sont appliqués **directement dans les requêtes SQL** pour une performance optimale
3. Les résultats incluent automatiquement les métadonnées de classification

### 4. Enrichissement des outils RAG existants

**Fichier**: `src/agents/rag_tool.py`

Les outils existants ont été enrichis avec de nouveaux paramètres optionnels :

#### `query_rag` enrichi avec :
- `content_type` : Filtre par type de contenu ('demand', 'section_header', 'regular')
- `section_filter` : Filtre par sections (['synthesis', 'demands', 'observations', etc.])
- `demand_priority` : Filtre par priorité de demande (1 ou 2)
- `count_only` : Mode comptage pour obtenir des statistiques

#### `query_rag_from_id` enrichi avec :
- Inclusion automatique des métadonnées de classification dans les résultats

## Utilisation

### 1. Appliquer la migration

```sql
-- Exécuter le script SQL pour ajouter les nouvelles colonnes
psql -d votre_base -f src/migrations/add_content_classification_to_rag.sql
```

### 2. Ré-indexer les documents existants

Les documents déjà indexés devront être ré-indexés pour bénéficier de la classification :

```python
from rag_system import RAGSystem

rag = RAGSystem()
# Ré-indexer un document
rag.index_document("chemin/vers/lettre.pdf")
```

### 3. Utiliser les outils enrichis

```python
# Recherche normale (maintenant enrichie avec métadonnées)
result = query_rag(
    keywords=["radioprotection"],
    source_names=["INSSN-LYO-2021-0469"]
)
# Les résultats incluent maintenant content_type, section_type, demand_priority

# Rechercher uniquement des demandes prioritaires
result = query_rag(
    keywords=["sûreté", "criticité"],
    source_names=["INSSN-LYO-2021-0469"],
    content_type="demand",
    demand_priority=1
)

# Compter les demandes à Blayais depuis 2024
stats = query_rag(
    keywords=[],  # Pas de filtrage par mots
    source_query="SELECT name FROM public.public_data WHERE site_name = 'Blayais' AND date >= '2024-01-01'",
    content_type="demand",
    demand_priority=1,
    count_only=True
)
# Retourne: {"total_count": 45, "by_document": {...}, "by_section": {...}, "by_priority": {...}}

# Recherche dans des sections spécifiques
result = query_rag(
    keywords=["constat"],
    section_filter=["observations", "synthesis"]
)

# Récupérer des blocs par ID (avec métadonnées)
blocks = query_rag_from_id(
    block_indices=[123, 124, 125],
    source_name="INSSN-LYO-2021-0469"
)
# Les blocs incluent maintenant les métadonnées de classification
```

## Avantages de l'intégration

1. **Unification** : Plus besoin de maintenir deux systèmes séparés (tables demands/letters ET rag_document_blocks)
2. **Rétrocompatibilité** : Les utilisations existantes des outils RAG continuent de fonctionner
3. **Contexte enrichi** : Les demandes sont searchables avec leur contexte complet
4. **Flexibilité** : Possibilité de filtrer par type de demande/section avec les mêmes outils
5. **Performance optimale** : 
   - Filtrage au niveau base de données (pas en Python)
   - Une seule requête au lieu de multiples requêtes
   - Utilisation des index PostgreSQL pour des recherches rapides
6. **Évolutivité** : Facile d'ajouter de nouveaux types de classification
7. **Simplicité** : Pas besoin d'apprendre de nouveaux outils, juste de nouveaux paramètres

## Notes importantes

- La détection fonctionne uniquement sur les documents qui correspondent aux patterns de lettres de suite
- Les documents non-lettres restent classifiés comme contenu "regular"
- La classification est faite lors de l'indexation, pas à la volée
- Compatible avec les deux backends PDF (PyMuPDF et NLM-Ingestor)

## Architecture technique

### Flux de données

1. **Indexation** : PDF → Classification → Base de données avec métadonnées
2. **Recherche** : Requête avec filtres → SQL optimisé avec WHERE clauses → Résultats enrichis

### Performance

L'approche choisie optimise les performances :
- **Filtrage SQL** : Les filtres `content_type`, `section_filter`, et `demand_priority` sont intégrés directement dans la clause WHERE
- **Une seule requête** : Les métadonnées sont retournées dans la même requête que les résultats de recherche
- **Index optimisés** : Utilisation d'index composites pour les requêtes filtrées fréquentes
