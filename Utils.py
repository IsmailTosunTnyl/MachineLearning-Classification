import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np
from sklearn.model_selection import learning_curve


class RawData:
    def __init__(self):
        # Define column names
        self.col_names = ["instance_id", "artist_name", "track_name", "popularity", "acousticness", "danceability",
                          "duration_ms",
                          "energy",
                          "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo",
                          "obtained_date",
                          "valence",
                          "music_genre"
                          ]

        # Define features
        self.feature_cols = ["popularity", "acousticness", "danceability", "duration_ms", "energy", "instrumentalness",
                             "key",
                             "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]

        # Define mapping values
        self.map_key = {"A": 1, "A#": 2, "B": 3, "B#": 4, "C": 5, "C#": 6,
                        "D": 7, "D#": 8, "E": 9, "E#": 10, "F": 11, "F#": 12, "G": 13, "G#": 14}
        self.map_mode = {"Minor": 0, "Major": 1}
        self.map_genres = {'Electronic': 0, 'Anime': 1, 'Jazz': 2, 'Alternative': 3, 'Country': 4, 'Rap': 5, 'Blues': 6,
                           'Rock': 7,
                           'Classical': 8, 'Hip-Hop': 9}

        # Access data file via Pandas lib and combine with column names
        self.df = pd.read_csv("music_genre.csv", header=None, names=self.col_names)

        # Cleaning the data to remove unwanted data
        self.df = self.df[self.df.tempo != "?"]
        self.df = self.df[self.df.duration_ms != -1]
        self.df = self.df[self.df.instrumentalness != 0]
        self.df.dropna()
        self.df.drop_duplicates()
        self.df.reset_index(drop=True)

        # Applying Mapping
        self.df["key"] = self.df["key"].map(self.map_key)
        self.df["mode"] = self.df["mode"].map(self.map_mode)
        self.df["music_genre"] = self.df["music_genre"].map(self.map_genres)

        # self.df = self.df.sample(frac=1).reset_index(drop=True)  # randomize data if necessary

        # drop empty rows and reindex
        self.df.dropna(inplace=True)


class Plotter:

    def plot_cofusion_matrix(self, y_test, y_pred, name):
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        class_names = ["y_test", "y_pred"]  # name  of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # create heatmap
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.subplots_adjust(left=0.086, top=0.917)
        plt.title(f'Confusion matrix for a {name}  model', fontsize=14, y=1.03)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()

    def plot_traning_curves(self, x, y, model, name):
        train_sizes = [1, 100, 500, 2000, 5000, 7654, 10000, 15000, 20000, 22000]
        train_sizes, train_scores, validation_scores = learning_curve(
            estimator=model,
            X=x,
            y=y, train_sizes=train_sizes, cv=5,
            scoring='neg_mean_squared_error')

        train_scores_mean = -train_scores.mean(axis=1)
        validation_scores_mean = -validation_scores.mean(axis=1)

        plt.style.use('seaborn')
        plt.plot(train_sizes, train_scores_mean, label='Training error')
        plt.plot(train_sizes, validation_scores_mean, label='Validation error')
        plt.ylabel('MSE', fontsize=14)
        plt.xlabel('Training set size', fontsize=14)
        plt.title(f'Learning curves for a {name} model', fontsize=18, y=1.03)
        plt.legend()
        plt.ylim(0, 40)
        plt.show()
