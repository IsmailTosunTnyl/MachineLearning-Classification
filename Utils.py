import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np

class RawData:
    def __init__(self):


        self.col_names = ["instance_id", "artist_name", "track_name", "popularity", "acousticness", "danceability",
                          "duration_ms",
                          "energy",
                          "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo",
                          "obtained_date",
                          "valence",
                          "music_genre"
                          ]
        self.feature_cols = ["popularity", "acousticness", "danceability", "duration_ms", "energy", "instrumentalness",
                             "key",
                             "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]

        self.map_key = {"A": 1, "A#": 2, "B": 3, "B#": 4, "C": 5, "C#": 6,
                        "D": 7, "D#": 8, "E": 9, "E#": 10, "F": 11, "F#": 12, "G": 13, "G#": 14}
        self.map_mode = {"Minor": 0, "Major": 1}
        self.map_genres = {'Electronic': 0, 'Anime': 1, 'Jazz': 2, 'Alternative': 3, 'Country': 4, 'Rap': 5, 'Blues': 6,
                           'Rock': 7,
                           'Classical': 8, 'Hip-Hop': 9}

        self.df = pd.read_csv("music_genre.csv", header=None, names=self.col_names)

        self.df.drop("obtained_date",inplace=True,axis=1)


        self.df = self.df[self.df.tempo != "?"]
        self.df = self.df[self.df.duration_ms != -1]
        self.df = self.df[self.df.instrumentalness != 0]
        self.df.dropna()
        self.df.reset_index(drop=True)
        self.df["key"] = self.df["key"].map(self.map_key)
        self.df["mode"] = self.df["mode"].map(self.map_mode)
        self.df["music_genre"] = self.df["music_genre"].map(self.map_genres)

        self.df.drop_duplicates()

        #self.df = self.df.sample(frac=1).reset_index(drop=True)  # randomize

        self.df.dropna(inplace=True)


class Plotter:

    def plot(self,y_test,y_pred):
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        class_names = [0, 1]  # name  of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # create heatmap
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()