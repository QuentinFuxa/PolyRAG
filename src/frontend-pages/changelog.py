import streamlit as st

st.header("Nouveautés :")
st.subheader("16/06/2025")
st.markdown("""
- Déploiement aux utilisateurs
""")

st.subheader("18/06/2025")
st.markdown("""
- Meilleure gestion des feedbacks
- Seulement la date de derniere maj de la conversation est affichée au survol d'une conversation
- Amélioration du modèle lors de certaines questions
- Ajout de la date limite de la phase de test du PoC (11/07/2025)
""")


st.subheader("23/06/2025")
st.markdown("""
### Questions testées et corrigées

| Question/Requête | Problème identifié | Nouvelle version | Statut |
|---|---|---|---|
| Donne moi la récurrence des écarts dans les LDS 2024 thème curiethérapie | Que 8 écarts sur ce thème | 40 maintenant | ✅ Corrigé |
| Fais un graphique du type d'écarts | Fonctionne pas | Affiche un graphique "Camembert" | ✅ Corrigé |
| Combien de centre de curiethérapie traite les cicatrices chéloides | Ne trouve pas | 1 centre (c'est le cas ?) | ✅ Corrigé |
| Combien de service de medecine nucléaire utilise le lutétium | Ne comprend pas le terme lutétium | Il y a 55 services de médecine nucléaire qui utilisent le lutétium | ✅ Corrigé |
| fais une synthèse sur les libéraux employeurs en 2024 | Ne trouve pas | ❌ | ❌ Non résolu : Ne comprends pas libéraux employeurs |
| fais un point sur la dosimétrie patient sur les PIR en 2024 | ne trouve rien | Trouve des résultats | ✅ Corrigé |
| combien d'établissement utilise les prestations de physiques médicales en 2024 | ne trouve rien | 626 établissements | ✅ Corrigé |
| monographie des LDS 2024 OARP | Ne trouve pas | ❌ | ❌ Non résolu : Ne comprends pas OARP - fournir un outil d'accès au lexique ? |
| fais une synthèse des LDS 2024 Organismes agréés pour la mesure du radon | ne détecte qu'une seule inspection en 2024 | ❌ | ✅ Corrigé |
| fais une monographie 2024 des inspections recherche Npx | ne fais de synthèse mais une liste | ⚠️ donne les demandes types les plus récurrentes, mais pas à proprement dit une synthèse | ⚠️ Partiellement corrigé |
| fais une monographie des LDS 2024 en industrie Npx | identification d'une seule inspection | Trouve 877 résultats | ✅ Corrigé |
| Affiche le nombre de demandes en LUDD depuis début 2025 | Pas le bon nombre | Bon nombre | ✅ Corrigé |
| Combien de LDS avec la thématique radioprotection en 2024 et 2025 en LUDD | N'en trouve pas | Trouve des inspections | ✅ Corrigé |
| Combien d'inspection fraude en LUDD en 2024 et 2025 | N'en trouve pas | ⚠️ En trouve 111, Réporting en trouve 17 avec la thématique fraude | ⚠️ Écart avec reporting. Limitation car pas d'accès aux thèmes |
| Nombre de demande dans la dernière inspection qu'il y a eu au GANIL | Ne trouve pas | La dernière inspection au GANIL a donné lieu à 9 demandes | ✅ Corrigé |
| Nombre de demande pour le GANIL en 2024 | Ne trouve pas | Il y a eu 24 demandes pour le GANIL en 2024 | ✅ Corrigé |
| Fais un bilan des écarts les plus fréquents en CNPE en 2024 | Ne trouve pas | Fonctionne bien | ✅ Corrigé |
| Quels sont les thèmes les moins inspectés en 2024 dans les CNPE par rapport à la période 2020 – 2023 | Ne trouve pas | ⚠️ Resultats mais pas forcément de bonne qualité : L'outil n'a pas accès aux thèmes | ⚠️ Limitation accès données non publiques |
| Quels sont les points communs entre les 10 dernières inspections de revue | Ne trouve pas | ❌ N'a pas accès aux métadonnées du type d'inspection. Fourni des résultats mentionnant "revu", mais c'est incorrect | ❌ Limitation accès données non publiques |
| Fais une synthèse 2024 sur les inspections sur évènement | Ne trouve pas | ❌ Ne trouve rien : N'a pas accès aux métadonnées du type d'inspection | ❌ Limitation accès données non publiques |
| fais une synthèse des inspections sur les traitements biocides | Ne trouve pas | Fonctionne bien | ✅ Corrigé |
| fais une synthèse sur la maîtrise de la réactivité en salle de commande | Ne trouve pas | Fonctionne bien | ✅ Corrigé |

## Légende des statuts
- ✅ **Corrigé** : Problème résolu, fonctionnalité opérationnelle
- ⚠️ **Partiellement corrigé** : Amélioration mais limitations subsistantes
- ❌ **Non résolu** : Problème persistant ou limitation système

### Autres améliorations
- Améliore l'affichage des Lettres de suite dans la page de conversations
- Objectifs remontés dans la page d'aide

""")