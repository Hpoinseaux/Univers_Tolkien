
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def main():
    st.image("6)image_streamlit/intro.jpg", use_column_width=True)
    center_title("Projet Tolkien")
    
    st.sidebar.image("6)image_streamlit/jrr.jpg")
    st.sidebar.header("Sommaire")
    sections = ["Introduction", "Analyse des textes", "Analyse de sentiments", "Détection de communauté", "Conclusion"]
    selected_section = st.sidebar.radio("", sections)

    if selected_section == "Introduction":
        introduction()
    elif selected_section == "Analyse des textes":
        analyse_textes()
    elif selected_section == "Analyse de sentiments":
        analyse_sentiments()
    elif selected_section == "Détection de communauté":
        detection_communaute()
    elif selected_section == "Conclusion":
        conclusion()

def center_title(title):
    st.markdown(f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)
def introduction():
    st.header("Plongée dans l’univers de Tolkien : une exploration fascinante à travers l’intelligence artificielle")
    st.write("""
    **Sommaire :**
    - [Introduction](#introduction)
    - [Analyse des textes](#analyse-des-textes)
    - [Analyse de sentiments](#analyse-de-sentiments)
    - [Détection de communauté](#détection-de-communauté)
    - [Conclusion](#conclusion)
    
   Bienvenue dans le Projet Tolkien, une initiative ambitieuse qui vous invite à découvrir les richesses cachées des œuvres de J.R.R. Tolkien, à travers une analyse profonde et innovante. Ce projet, qui allie littérature et technologie, vous propose une nouvelle façon de lire et d’interpréter les textes de l’auteur du Seigneur des Anneaux.

    Nous commencerons par vous offrir une analyse détaillée d’un extrait clé de son œuvre, avant de vous dévoiler les algorithmes d’intelligence artificielle que nous avons mis en place pour répondre à des questions complexes liées à ces textes. Enfin, nous partagerons avec vous les enseignements personnels tirés de cette aventure intellectuelle et technologique.

    Rejoignez-nous dans ce voyage où la magie de Tolkien rencontre la puissance de l'IA, pour une expérience de lecture et d’analyse inédite !
    """)
  
       
def analyse_textes():
    st.header("Analyse des textes")
    st.write("""
    Dans un premier temps, nous avons réalisé une analyse approfondie des textes de Tolkien afin d'identifier les questions pertinentes que l’on pourrait explorer et de déterminer en quoi les techniques de traitement du langage naturel (NLP) pouvaient nous aider à y répondre. Les œuvres sur lesquelles nous avons concentré notre étude sont les suivantes :

    The Hobbit (1937)
    The Lord of the Rings (1954-1955)
    The Silmarillion (1977)
    Unfinished Tales (1980)
    The Children of Húrin (2007)
    The Fall of Gondolin (2018)
    The Letters of J.R.R. Tolkien (1981)

    Pour mener à bien cette analyse, nous avons utilisé des outils tels que Spacy et nltk pour le traitement du texte, ainsi que Wordcloud et matplotlib pour la création de visuels permettant de mieux illustrer nos résultats.

    Voici les premières conclusions issues de cette analyse :
    """)
     # Affichage des images en fonction du choix de l'utilisateur
    image_choice = st.radio("Choisissez une image :", ["Anneau", "Arc","Hache","Gandalf"])

    if image_choice == "Anneau":
        st.image("6)image_streamlit/wc_100mot_ring.png", use_column_width=True)
    elif image_choice == "Arc":
        st.image("6)image_streamlit/wc_100mots_arc.png", use_column_width=True)
    elif image_choice == "Hache":
        st.image("6)image_streamlit/wc_50NP_all.png", use_column_width=True)
    elif image_choice == "Gandalf":
        st.image("6)image_streamlit/wc_lotr_50mots.png", use_column_width=True)    
    
    image_choice = st.selectbox("Choisissez un graphe de tous les livres :", ["Histogramme occurence types des mots par livre", "occurence type des mots"])
    if image_choice == "Histogramme occurence types des mots par livre":
        st.image("6)image_streamlit/hist_type_par_livre.png", use_column_width=True)
    elif image_choice == "occurence type des mots":
        st.image("6)image_streamlit/hist_type_all.png", use_column_width=True)
    
    st.write("Tableau occurences noms par livre:")
    df = pd.read_csv("6)image_streamlit/noms.csv")  
    def obtenir_lignes_au_hasard():
        return df.sample(15)
    if st.button("Changer les 15 noms aléatoires"):
        nouvelles_lignes = obtenir_lignes_au_hasard()
        st.write(nouvelles_lignes)
    else:
        st.write(obtenir_lignes_au_hasard())
        
    image_choice = st.selectbox("Choisissez un graphe LOTR :", ["occurence lieux", "occurence race","frequence noms propres"])
    if image_choice == "occurence lieux":
        st.image("6)image_streamlit/graphe_occ_lieux.png", use_column_width=True)
    elif image_choice == "occurence race":
        st.image("6)image_streamlit/graphe_occ_race.png", use_column_width=True)
    elif image_choice == "frequence noms propres":
        st.image("6)image_streamlit/hist_freq_30NP_lotr.png", use_column_width=True)
    
    audio_file = "6)image_streamlit/A-vrai-dire-il-y-a-bien-une-chose.mp3"
    st.audio(audio_file, format='audio/mp3', start_time=0)
    
    

def analyse_sentiments():
    st.header("Analyse de sentiments")
    st.write("""
    Dans cette section, nous allons explorer l’utilisation du traitement du langage naturel (NLP) pour réaliser une analyse des sentiments des personnages principaux du Seigneur des Anneaux (LOTR) et du Hobbit.

    En nous appuyant sur nos connaissances des œuvres, nous savons que les personnages de ces deux récits ne sont pas uniquement positifs ou négatifs. Ils traversent fréquemment des périodes sombres avant de connaître des moments de rédemption, reflétant ainsi une évolution complexe de leurs émotions.

    Dans un premier temps, nous allons tester si l’analyse des sentiments peut effectivement identifier si un personnage est perçu comme étant plutôt positif ou négatif, selon le contexte du texte.

    Voici les premiers résultats de cette analyse :
    """)
    st.write("Comparatif Spacy et NLTK:")
    image_path1 = "6)image_streamlit/graphe_pourcentage_sent_nltk.png"
    image_path2 = "6)image_streamlit/graphe_pourcentage_sent_spacy.png"
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_path1, caption="NLTK", use_column_width=True)
    with col2:
        st.image(image_path2, caption="Spacy", use_column_width=True)
    st.write("""
    Les deux tableaux montrent que Spacy tend à classer certains personnages initialement perçus comme négatifs parmi les personnages positifs, tandis que nltk se révèle plus efficace pour identifier les termes négatifs. Nous remarquons également que les sentiments des personnages restent souvent assez nuancés, avec une différence d’environ 10 % entre les sentiments négatifs et positifs. De plus, une proportion importante de sentiments neutres apparaît, représentant environ 40 % des évaluations pour chaque personnage. Cela suggère que l’algorithme rencontre encore des difficultés à distinguer clairement les sentiments positifs et négatifs, peut-être en raison du style d’écriture plus ancien de J.R.R. Tolkien, qui pourrait influencer l’interprétation des sentiments.
    """)
    
    st.write("Evolution des sentiments par chapitres avec NLTK:")
    show_image_1 = st.checkbox("LOTR")
    show_image_2 = st.checkbox("The Hobbit")

    if show_image_1:
        with open("6)image_streamlit/characters_sentiments.pkl", "rb") as file:
            characters_sentiments = pickle.load(file)
        selected_characters = st.multiselect("Choisissez les personnages à afficher", list(characters_sentiments.keys()))
        plt.figure(figsize=(12, 8))
        for character in selected_characters:
            sentiments = characters_sentiments.get(character)
            if sentiments is not None:
                plt.plot(sentiments, label=character)

        plt.title("Évolution des sentiments pour différents personnages par 5 chapitres")
        plt.xlabel("Evolution durant les livres.")
        plt.ylabel("Score de sentiment moyen")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

    if show_image_2:
        with open("6)image_streamlit/characters_sentiments_bilbo.pkl", "rb") as file:
            characters_sentiments = pickle.load(file)
        selected_characters = st.multiselect("Choisissez les personnages à afficher", list(characters_sentiments.keys()))
        plt.figure(figsize=(12, 8))
        for character in selected_characters:
            sentiments = characters_sentiments.get(character)
            if sentiments is not None:
                plt.plot(sentiments, label=character)

        plt.title("Évolution des sentiments pour différents personnages par chapitres")
        plt.xlabel("Evolution durant les livres.")
        plt.ylabel("Score de sentiment moyen")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)
    


def detection_communaute():
    st.header("Détection de communauté")
    st.write("""
    Dans cette section, nous avons créé plusieurs graphiques pour analyser la manière dont les algorithmes de détection de communautés identifient les groupes au sein des textes. Pour ce faire, nous avons utilisé la vectorisation des mots avec Word2Vec et la génération de graphes à l’aide de NetworkX.

    Dans un premier temps, nous avons réalisé un graphe de détection de communautés basé sur l’œuvre Le Seigneur des Anneaux (LOTR) :
    """)
    images_to_display = st.multiselect("Choisissez un graphe :", ["Louvain", "Fruchtmann & Reynolds"])

    for image in images_to_display:
        if image == "Louvain":
            st.image("6)image_streamlit/louvain_lotr.png", use_column_width=True)
        elif image == "Fruchtmann & Reynolds":
            st.image("6)image_streamlit/freuchtmann_lotr.png", use_column_width=True)

    if st.button("Afficher graphe LOTR 3D"):
        with open('6)image_streamlit/figure_3dlotr.pickle', 'rb') as file:
            loaded_fig = pickle.load(file)
            st.plotly_chart(loaded_fig)

    st.write("""
    Les graphes que nous avons obtenus, bien que relativement similaires, montrent que certaines communautés semblent se regrouper en fonction de caractéristiques comme la race (par exemple, les hobbits) ou des liens récurrents dans l'intrigue (comme l'amitié entre Gimli et Legolas). L'algorithme a réussi à identifier trois grandes communautés, mais ces groupes ne correspondent pas nécessairement à une séparation par race ou à une proximité évidente dans l'histoire. La détection des communautés reste donc complexe. Cependant, nous remarquons que les personnages principaux se situent généralement au centre du graphe de Fruchterman-Reingold, ce qui témoigne d'une reconnaissance des personnages clés, sans pour autant établir de liens directs entre eux, les plaçant ainsi dans deux communautés distinctes.

    Nous avons ensuite exploré la détection des communautés pour l'ensemble des personnages à l’aide de la méthode Louvain :
    """)
    if st.button("Afficher graphe de tous les livres"):
        st.image("6)image_streamlit/louvain_all.png", use_column_width=True)
    if st.button("Afficher graphe de tous les livres 3D"):
        with open('6)image_streamlit/figure_3dtolkien.pickle', 'rb') as file:
            loaded_fig = pickle.load(file)
            st.plotly_chart(loaded_fig) 
    st.write("""
    Dans ce graphe, nous avons limité l’analyse aux 20 personnages les plus récurrents de chaque livre. Il apparaît que l’algorithme a détecté trois communautés, dont une est faiblement représentée et l’autre surreprésentée. Cela suggère des difficultés à établir des liens entre les communautés, que ce soit par race, époque ou livre. Cependant, on remarque que les groupes les plus distincts sont ceux des races humaines et des hobbits, représentées en vert sur la droite du graphe, qui forment des clusters plus compacts.

    Par la suite, nous avons également exploré l’analyse des communautés par lieux en utilisant la méthode Louvain :
    """)
    if st.button("Afficher graphe lieux"):
        st.image("6)image_streamlit/louvain_lieux_lotr.png", use_column_width=True)

    
def conclusion():
    st.header("Conclusion")
    audio_file = "6)image_streamlit/Fuyez-pauvres-fous!.mp3"
    st.audio(audio_file, format='audio/mp3', start_time=0)
    st.write("""
    Dans un premier temps, l’analyse de sentiments a montré une capacité intéressante à suivre l’évolution des émotions des personnages, souvent fluctuantes, avec une phase de rédemption avant leur disparition dans l’écriture. Toutefois, l’algorithme rencontre des difficultés à catégoriser les personnages comme strictement négatifs ou positifs, ce qui pourrait s'expliquer par le style descriptif de Tolkien, qui tend à rendre ses personnages plus nuancés et complexes.

    Il serait peut-être pertinent de combiner cette analyse avec les phrases suivantes pour affiner les résultats ou, à l’inverse, de supprimer les phrases trop neutres ou descriptives. Une autre option pourrait être de créer un modèle d'analyse spécifique, basé sur un dictionnaire de mots propres à l'univers de Tolkien, afin d'apporter plus de précision et de pertinence à l’analyse.

    En ce qui concerne la détection de communautés, nos résultats n'ont pas permis d’aboutir à des conclusions claires et exploitables. D’après notre expertise, une analyse plus fine devrait révéler une plus grande diversité de communautés, mieux définies. Il est donc évident que les outils d’analyse de texte, dans leur état actuel, nécessitent encore du travail pour atteindre une précision suffisante.

    Cette exploration a néanmoins révélé une vérité fondamentale : pour mener une analyse véritablement pertinente, il est crucial de bien comprendre le domaine d’étude. Cette connaissance permet de faire les choix les plus adaptés pour répondre aux questions les plus profondes. Nous avons également pris conscience de l'importance de l’analyse et de l’expertise humaines dans l’interprétation des résultats. L'humain reste indispensable pour affiner les paramètres et tirer des enseignements significatifs des données.

    Ainsi, au fil de ces étapes, nous nous retrouvons face à un univers de possibilités infinies, semblable à celui de Tolkien : un monde où l'intelligence artificielle, à l’instar des personnages de la Terre du Milieu, traverse des épreuves et des rédemptions, toujours en quête d’un équilibre entre la machine et l’humanité. Peut-être qu’en approfondissant nos recherches, nous pourrons un jour donner à nos algorithmes une compréhension aussi profonde et nuancée des émotions humaines que celle de l’auteur lui-même.
""")
    st.image("6)image_streamlit/fin.jpg", use_column_width=True)
    audio_file = "6)image_streamlit/Adieu-Ne-vous-detournez-pas-de-votre-but.mp3"
    st.audio(audio_file, format='audio/mp3', start_time=0)

st.markdown("<h3 style='text-align: right; font-size: 14px;'>Hadrien Poinseaux</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()