import streamlit as st
from auth_helpers import ensure_authenticated
from display_texts import dt

st.set_page_config(page_title=dt.SUGGESTIONS_GALLERY_TITLE, page_icon=":material/lightbulb:")

if dt.LOGO:
    st.logo(image=dt.LOGO, size="large")
    custom_css = """
    <style>
        div[data-testid="stSidebarHeader"] > img, div[data-testid="collapsedControl"] > img {
            height: 4rem;
            width: auto;
        }
        div[data-testid="stSidebarHeader"], div[data-testid="stSidebarHeader"] > *,
        div[data-testid="collapsedControl"], div[data-testid="collapsedControl"] > * {
            display: flex;
            align-items: center;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

if not ensure_authenticated():
    st.stop()

st.header("📍 Définition du périmètre")
st.write("""
L’assistant SIANCE utilise de l’IA générative (Mistral Large) sur l’historique des lettres de suite disponibles sur le site de l’ASN (jusqu’à fin 2025) et les documents uploadés par l’utilisateur (documents publics uniquement).
Il permet d’interagir avec l’ensemble des documents et utilise l’ensemble des questions ainsi que leurs contextes dans une même conversation.
""")

# Section : Exemples de cas d’usage
st.header("🧪 Exemples de cas d’usage")
use_cases = [
    "Fais-moi une **synthèse monographique** des demandes prioritaires depuis début 2025",
    "Donne-moi la lettre de suite **la plus récente** mentionnant les télécommunications de crise",
    "Fais-moi un **résumé** des dernières LDS sur Chooz ?",
    "Montre **l'évolution comparative** du nombre d'inspections par mois entre Tricastin et Gravelines depuis 2022",
    "Résumé de la **dernière inspection** à Chinon",
    "Fais-moi un **bilan** de la situation 2025 sur Golfech",
    "Affiche une **table** du nombre de LDS par année dans le NPX",
    "**A-t-on observé** des LDS qui parlent de fraude en 2024 ?",
    "Donne-moi un résumé des **dernières LDS fournisseurs**"
]
for uc in use_cases:
    st.markdown(f"- {uc}")

# Section : Objectifs du POC
st.header("🎯 Objectifs du POC")
objectifs = [
    "Valider la valeur métier de ce type de solution",
    "Définir les cas d’usages les plus intéressants",
    "Identifier les axes d’amélioration et les freins éventuels"
]
for obj in objectifs:
    st.markdown(f"- {obj}")

# Section : Réagir aux réponses
st.header("💬 Réagir aux réponses de l’assistant")

st.subheader("Que faire si la réponse ne convient pas ?")
reactions = {
    "Demander une précision ou une reformulation": "Peux-tu détailler uniquement les demandes liées au risque incendie ?",
    "Indiquer les éléments manquants ou incohérents": "Il manque la date de l’inspection dans la réponse",
    "Demander une source ou à afficher le passage dans le PDF": "Montre-moi l’extrait dans la lettre de suite"
}
for k, v in reactions.items():
    st.markdown(f"- **{k}**  \n  _Ex : « {v} »_")

st.subheader("Exemples d’améliorations possibles :")
st.markdown("""
- “La synthèse n’est pas claire, peux-tu la résumer en 3 points ?”
- “Tu as oublié le site concerné, peux-tu le préciser ?”
""")

# Section : Prise en main
st.header("🧪 Que tester pendant la phase de prise en main ?")
tests = [
    "Faire une synthèse d’un site ou d’une thématique sur plusieurs années",
    "Chercher une demande précise dans une lettre",
    "Comparer plusieurs sites ou plusieurs années",
    "Demander le nombre total de demandes prioritaires/autres demandes sur une période",
    "Afficher le PDF avec surlignage pour vérifier la conformité de la réponse",
    "Tester avec des mots-clés inhabituels (ex : “procédure d’urgence”, “télécommunication”)",
    "Utiliser la fonction de statistiques pour obtenir des chiffres globaux",
    "Observer la robustesse de l’assistant si vous faites des fautes de frappe ou posez des questions vagues"
]
for t in tests:
    st.markdown(f"- {t}")

# Section : Feedback
st.header("📝 Remplir le feedback et donner un avis utile")

st.subheader("Comment remplir la notation et le commentaire :")
st.markdown("""
- **Notez l’expérience de 1 à 5 étoiles**  
  1 = Pas du tout satisfait, 5 = Très satisfait
""")

st.subheader("Commentez en quelques phrases :")
st.markdown("""
- Ce qui a bien fonctionné (clarté, rapidité, précision…)
- Ce qui doit être amélioré (résultats peu pertinents, manque de détails…)
- Les bugs ou erreurs rencontrés
- Suggestions de nouvelles fonctionnalités
""")

st.subheader("Signalez les cas où vous n’avez pas obtenu ce que vous attendiez :")
st.markdown("_Ex : « La synthèse ne répondait pas à ma question, il manquait les actions demandées. »_")


st.header("Nouveautés :")
st.subheader("12/06/2025")
st.markdown("""
- Correction de bugs mineurs
- Correction de l'upload de documents PDF
""")