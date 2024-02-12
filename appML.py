import pandas as pd
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

st.set_option("deprecation.showPyplotGlobalUse", False)


def main():
    st.title(' Credit card fraud detection with Streamlit')
    st.subheader('Author : Harrabi Fadwa')

    # Fonction d'importation des données
    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('creditcard.csv')
        return data

    # Affichage de la table de données
    df = load_data()
    df_sample = df.sample(100)
    if st.sidebar.checkbox('View raw data', False):
        st.subheader('“Creditcard” dataset: Sample of 100 observations')
        st.write(df_sample)

    seed = 123

    # Fonction de division en ensembles d'entraînement et de test
    @st.cache_data(persist=True)
    def split(df):
        x = df.drop(columns='Class', axis=1)
        y = df['Class']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=seed)
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = split(df)

    class_names = ['T .Authentique', 'T .Frauduleuse']

    classifier = st.sidebar.selectbox(
        'Classificateur',
        ('Random Forest', 'SVM', 'Logistic Regression')
    )

    def plot_metrics(graphes):
        if "Confusion Matrix" in graphes:
            st.subheader("Matrice de confusion")
            cm = confusion_matrix(y_test, y_pred)
            st.write("Confusion Matrix :")
            plt.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
            plt.colorbar()
            plt.xlabel('Prediction')
            plt.ylabel('Real label')
            st.pyplot()

        if "ROC Curve" in graphes:
            st.subheader("Courbe ROC")
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
            plt.plot(fpr, tpr, label="ROC Curve")
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.title("Courbe ROC")
            plt.legend(loc="lower right")
            st.pyplot()

        if "Precision recall curve" in graphes:
            st.subheader("Courbe Précision-Rappel")
            precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(x_test)[:, 1])
            plt.plot(recall, precision, label="Precision-Recall Curve")
            plt.xlabel("Rappel (Recall)")
            plt.ylabel("Précision")
            plt.title("Courbe Précision-Rappel")
            plt.legend(loc="lower left")
            st.pyplot()

    # RandomForest
    if classifier == 'Random Forest':
        st.sidebar.subheader('Hyperparamètres du modèle')
        n_arbres = st.sidebar.number_input('Choisir le nombre d\'arbres dans la forêt', 100, 1000, step=10)
        profondeur_arbre = st.sidebar.number_input('Choisir la profondeur maximale de l\'arbre', 1, 20, step=1)
        bootstrap = st.sidebar.radio('Echantillons bootstrap?', ('True', 'False'))

        # Visualisation
        graph_per = st.sidebar.multiselect(
            "Choisir un graphique de performance du modèle ML",
            ("Confusion Matrix", "ROC Curve", "Precision recall curve")
        )

        if st.sidebar.button('Exécuter', key='classify'):
            st.subheader('Random Forest Results')
            model = RandomForestClassifier(
                n_estimators=n_arbres,
                max_depth=profondeur_arbre,
                bootstrap=(bootstrap == 'True')  # Convert to boolean
            )
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred).round(3)

            st.write("Précision:", precision.round(3))
            st.write('Rappel (Recall):', recall)
            st.write("Exactitude (Accuracy):", accuracy.round(3))

            # Affichage des graphiques
            plot_metrics(graph_per)

    # SVM
    if classifier == 'SVM':
        st.sidebar.subheader('Hyperparamètres du modèle SVM')
        C = st.sidebar.number_input('Paramètre de régularisation C', 0.01, 10.0, step=0.01)
        kernel = st.sidebar.selectbox('Type de noyau', ('linear', 'rbf', 'poly', 'sigmoid'))
        gamma = st.sidebar.number_input('Paramètre gamma (pour les noyaux rbf, poly, sigmoid)', 0.001, 1.0, step=0.001)

        # Visualisation
        graph_per = st.sidebar.multiselect(
            "Choisir un graphique de performance du modèle ML",
            ("Confusion Matrix", "ROC Curve", "Precision recall curve")
        )

        if st.sidebar.button('Exécuter', key='classify'):
            st.subheader('SVM Results')
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred).round(3)

            st.write("Précision:", precision.round(3))
            st.write('Rappel (Recall):', recall)
            st.write("Exactitude (Accuracy):", accuracy.round(3))

            # Affichage des graphiques
            plot_metrics(graph_per)

    # Logistic Regression
    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Hyperparamètres du modèle de Régression Logistique')
        C = st.sidebar.number_input('Paramètre de régularisation C', 0.01, 10.0, step=0.01)
        penalty = st.sidebar.radio('Type de pénalité', ('l1', 'l2'))
        max_iter = st.sidebar.number_input('Nombre maximal d\'itérations', 100, 10000, step=100)

        # Visualisation
        graph_per = st.sidebar.multiselect(
            "Choisir un graphique de performance du modèle ML",
            ("Confusion Matrix", "ROC Curve", "Precision recall curve")
        )

        if st.sidebar.button('Exécuter', key='classify'):
            st.subheader('Logistic Regression Results')
            model = LogisticRegression(C=C, penalty=penalty, max_iter=max_iter)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred).round(3)

            st.write("Précision:", precision.round(3))
            st.write('Rappel (Recall):', recall)
            st.write("Exactitude (Accuracy):", accuracy.round(3))

            # Affichage des graphiques
            plot_metrics(graph_per)


if __name__ == '__main__':
    main()
