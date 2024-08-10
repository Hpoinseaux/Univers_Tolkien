
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
    st.header("Introduction")
    st.write("""
    **Sommaire :**
    - [Introduction](#introduction)
    - [Analyse des textes](#analyse-des-textes)
    - [Analyse de sentiments](#analyse-de-sentiments)
    - [Détection de communauté](#détection-de-communauté)
    - [Conclusion](#conclusion)
    
    Bienvenue dans le projet Tolkien. Ce projet explore différents aspects des œuvres de JRR Tolkien à travers diverses analyses et applications.
    
    Nous allons d'abord vous présenter notre analyse de texte. Ensuite, nous vous montrerons les divers algorithmes que nous avons utilisés pour répondre à certaines de nos questions. Enfin, nous conclurons sur l'apport que ce projet a eu sur notre enrichissement personnel.
    """)
  
       
def analyse_textes():
    st.header("Analyse des textes")
    st.write("""
    Dans un premier nous avons effectué une analyse de texte pour arriver à établir qu'elle était les questions que l'on pouvait se poser et en quoi le NLP pourrait analyser et répondre à nos questions. Nous nous sommes d'abord questionnés sur l'étude des différents textes qui sont :
    - The Hobbit (1937)
    - The Lord of the Rings (1954-1955)
    - The Silmarillion (1977)
    - Unfinished Tales (1980)
    - The Children of Húrin (2007)
    - The Fall of Gondolin (2018)
    - The Letters of J.R.R. Tolkien (1981)

    Pour cela nous nous sommes appuyés sur Spacy, nltk pour l'analyse de texte ,  Wordcloud et matplotlib pour les visuels.

    Voici nos premières analyses :
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
    
    
    

def analyse_sentiments():
    st.header("Analyse de sentiments")
    st.write("""
    Dans cette partie nous allons aborder le NLP afin de réaliser une analyse de sentiments sur les différents personnages de LOTR et du Hobbit.
    De part nos connaissances nous savons que les personnages de LOTR et du hobbit de sont pas toujours positifs ou négatifs. Ils passent souvent par des phases négatives puis redeviennent positives comme une forme de rédemption.

    Dans une première phase nous allons voir si l’analyse de sentiments arrive à détecter si un personnage est plutôt positif ou négatif par rapport au texte.
    Voici une première analyse :
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
    Sur ces deux tableaux, nous pouvons voir que Spacy met en avant des personnages censés être négatifs comme des personnages positifs. Nltk se montre lui plus performant pour repérer les mots plus négatifs. Nous constatons aussi que les personnages restent régulièrement assez mitigés avec une proximité entre les sentiments négatifs et positifs.( 10% d'écart environ au maximum). <on peut aussi constater de nombreux sentiments neutres(environ 40% par personnage), on peut donc émettre l'hypothèse que l'algorithme montre encore des difficultés à repérer les sentiments positif et négatif dans le style d'écriture de J.R.R Tolkien qui est plus ancien.
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
    Dans cette partie nous avons effectué divers graphes afin de savoir comment les algorithmes de détection de communauté allaient détecter les communautés en s’appuyant sur une vectorisation des mots avec  Word2vec et la création de graphe avec NetworkX.
 
    Dans un premier temps avons effectué un graphe avec une détection de communauté sur LOTR :
    """)
    images_to_display = st.multiselect("Choisissez un graphe :", ["Louvain", "Fruchtmann & Reynolds"])

    for image in images_to_display:
        if image == "Louvain":
            st.image("6)image_streamlit/louvain_lotr.png", use_column_width=True)
        elif image == "Fruchtmann & Reynolds":
            st.image("6)image_streamlit/freuchtmann_lotr.png", use_column_width=True)

    if st.button("Afficher graphe LOTR 3D"):
        try:
            # Assurez-vous que le chemin est correct
            with open('6)image_streamlit/figure_3dlotr.pickle', 'rb') as file:
                loaded_fig = pickle.load(file)

            # Si loaded_fig est du HTML ou contient du code HTML pour le rendu
            html_str = loaded_fig.to_html()  # Assurez-vous que `loaded_fig` a la méthode `to_html()`
            components.html(html_str, height=600)
        except FileNotFoundError:
            st.error("Le fichier figure_3dlotr.pickle est introuvable.")
        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
    st.write("""
    Dans ces graphes assez similaires, nous pouvons voir que parfois les communautés sont réunies par race (exemple : hobbit) ou par liens récurrents dans le livre (gimli et legolas). Elle a pu identifier 3 grandes communautés ce qui ne représente pas forcément une séparation par race ou de proximité dans le livre. La détection reste encore difficile. Nous constatons cependant que la plupart des personnages principaux sont situés au centre sur le graphe de fruchterman-reingold qui montre une reconnaissance des personnages principaux sans forcément faire des liens entre eux (séparé en deux communautés).
 
    Par la suite nous avons essayé de savoir la détection des communautés sur tous les personnages des livres avec la méthode Louvain :
    """)
    if st.button("Afficher graphe de tous les livres"):
        st.image("6)image_streamlit/louvain_all.png", use_column_width=True)
    if st.button("Afficher graphe de tous les livres 3D"):
        with open('6)image_streamlit/figure_3dtolkien.pickle', 'rb') as file:
            loaded_fig = pickle.load(file)
        loaded_fig.show()
    st.write("""
    Dans ce graphe nous avons limité aux 20 personnages les plus récurrents de chaque livre. Nous constatons aussi qu’il est resté sur la détection de 3 communautés avec une faiblement représenté et une surreprésenté. Il éprouve donc des difficultés à relier des communautés entre elles que ce soit par race, par période, par livre. Nous constatons quand même que là aussi ce sont les races qui ressortent le plus avec celle des humaines et des hobbits en vert sur la droite qui sont plus compacts.
  
    Par la suite nous avons aussi chercher par lieux avec la méthode Louvain :
    """)
    if st.button("Afficher graphe lieux"):
        st.image("6)image_streamlit/louvain_lieux_lotr.png", use_column_width=True)

    
def conclusion():
    st.header("Conclusion")
    st.write("""
    Dans un premier temps pour l’analyse de sentiments nous pouvons voir qu’elle repère assez bien l' évolution des émotions des personnages qui sont fluctuants avec une phase de rédemption avant leur disparition(dans l’écriture). Cependant elle éprouve des difficultés à inscrire des personnages plutôt négatif ou positif car selon notre hypothèse Tolkien fait de nombreuses description qui rend le personnage plus neutre. Il faudrait peut-être assemblé avec la phrase suivante pour voir si l’analyse est plus pertinente ou supprimer les phrases qui restent trop neutres donc descriptives. Une autre possibilité serait d'entraîner un modèle de dictionnaire spécifique sur les mots de Tolkien afin de rendre l’analyse plus pertinente
    Nous pouvons aussi dire que nos résultats de détection de communauté ne nous permettent pas d’avoir un résultat clair et exploitable car selon notre expertise il devrait avoir une plus grande possibilité de communauté et qu’elle soit mieux identifiée.
    Il est donc difficile d’exploiter nos outils d’analyse de texte en l’état, il faudrait un temps supplémentaire pour tester et approfondir nos démarches afin de pouvoir rendre l’algorithme plus précis.
    Au travers ses étapes nous avons pu constater dans un premier temps qu’il était important de bien connaître et comprendre le domaine dans lequel nous allons travailler afin d’être plus pertinents dans notre analyse et de faire les choix les plus appropriés afin de répondre  à des questions pertinentes . 
    Nous avons pu constater l’importance de l’analyse et l’expertise humaine sur les résultats  afin de mieux interpréter, d’évaluer et d’affiner certains paramètres pour la détection de communauté.
""")


if __name__ == "__main__":
    main()