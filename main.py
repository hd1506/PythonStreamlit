import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
st.title('Prime Medical Prediction')



st.set_option('deprecation.showPyplotGlobalUse', False)
sidebar_options=st.sidebar.selectbox(
    "Options",
    ("Presentation","Etude statistique","Classification","Prediction")
    )

data=pd.read_csv(r'C:\Users\setze\Desktop\Medicalpremium.csv')

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params


def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf


if sidebar_options == "Presentation":
    #st.title("Presentation")
    sidebar_options1=st.sidebar.selectbox(
    "Presentation",
    ("Presentation Generale du projet","Qui sommes-nous ?")
    )
    if(sidebar_options1=="Presentation Generale du projet"):
        st.title("Presentation Generale du projet")
        st.write("Notre projet s’articule autour de la prédiction des couts relatifs au recours aux services de santé pour différents types de profil selon plusieurs critères comme l’âge et les maladies chroniques. Ce projet s’inscrit dans une démarche de création d’un système décisionnelle intelligent permettant une symbiose entre les profils qui se présentent à l’assurance de recouvrement médicale et ces agences eux même.")
        st.write("Ainsi, notre projet permettra à l’agence d’assurance de prédire les couts qu’un bénéficiaire de l’assurance aura à débourser pour profiter pleinement des services médicaux et pourra ainsi distinguer les profils et les offres d’assurance")
    if(sidebar_options1=="Qui sommes-nous ?"):
        st.title("Qui sommes-nous ?")
        
        
        
        
if sidebar_options == "Etude statistique":
    #st.title("Etude statistique")
    
    sidebar_options2=st.sidebar.selectbox(
    "Statistiques",
    ("Ensemble d'etudes","Description donnees","Correlations","Graphes")
    )
    
    if(sidebar_options2=="Ensemble d'etudes"):
        st.title("Ensemble d'etudes")
        st.table(data.columns)
        
    if(sidebar_options2=="Description donnees"):
        st.title("Description donnees")
        st.table(data.describe())
        l=list(data.columns)
        sidebar_options2=st.sidebar.selectbox(
            "Features",
            (l[:-1])
            )
        fig,ax=plt.subplots(figsize=(12,6))
        for i in l[:-1]:
            if(sidebar_options2==i):
                sns.barplot(y=i,x='PremiumPrice',data=data,ax=ax)
                st.pyplot()
        
        
        
        
    if(sidebar_options2=="Correlations"):
        st.title("Correlations")
        st.table(data.corr().style.background_gradient())
        sns.jointplot(x='Height',y='Weight',data=data,height=8,kind='kde')
        st.pyplot()
    
    if(sidebar_options2=="Graphes"):
        st.title("Graphes")
        l=list(data.columns)
        sidebar_options2=st.sidebar.selectbox(
            "Features",
            (l[:-1])
            )
        for i in l[:-1]:
            if(sidebar_options2==i):
                sns.displot(x=i,data=data,kde=True)
                st.pyplot()
    


if sidebar_options == "Prediction":
    st.title("Prediction")
    st.write("under contruction XD")
    l=list(data.columns)
    age=st.text_input('Age','18')
    diabetes=st.text_input('Diabete','0')
    bloodpressure=st.text_input('Pression Sanguine','0')
    transplants=st.text_input('Transplantation','1')
    chronicdiseases=st.text_input('Maladie Chronique','1')
    taille=st.text_input('Taille','169')
    poids=st.text_input('Poids','68')
    allergies=st.text_input('Allergie','0')
    cancer=st.text_input('Cancer','0')
    operations=st.text_input('Chirurgie','1')
    boutton=st.button("Envoyer")
    if(boutton):
        import random
        a=random.randint(15000, 40000)
        st.write("Le prix potentiel est de ", a )

if sidebar_options == "Classification":
    
    
    
    st.write("""
             # Explore different classifier 
             Which one is the best?
             """)
    
    dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Breast Cancer', 'Wine')
    )
    
    st.write(f"## {dataset_name} Dataset")
    #st.write("Medicalpremium")


    classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest','MultiOutputClassifier')
    )
    
    
    X, y = get_dataset(dataset_name)
    
    #X = pd.read_csv(r'C:\Users\setze\Desktop\Medicalpremium.csv')
    #y = pd.read_csv(r'C:\Users\setze\Desktop\Medicalpremium.csv')
    st.write('Shape of dataset:', X.shape)
    st.write('number of classes:', len(np.unique(y)))



    params = add_parameter_ui(classifier_name)



#    knn=KNeighborsClassifier(n_neighbors=3)
#    clf = MultiOutputClassifier(knn, n_jobs=-1)
    clf = get_classifier(classifier_name, params)
    #st.write("coucou")
#### CLASSIFICATION ####

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    #st.write("coucou")
    clf.fit(X_train, y_train)
    #st.write("coucou")
    y_pred = clf.predict(X_test)
    #st.write("coucou")
    acc = accuracy_score(y_test, y_pred)
    #st.write("coucou55")
    st.write(f'Classifier = {classifier_name}')
    st.write(f'Accuracy =', acc)

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2,
                c=y, alpha=0.8,
                cmap='viridis')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()

#plt.show()
    st.pyplot(fig)
