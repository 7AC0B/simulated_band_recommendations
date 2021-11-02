import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import copy
import plotly.graph_objs as go
import plotly.express as px



class Band:
    def __init__(self, id, genre_clusters, band_deviation_from_genre, all_band_positionings, quietly=True):
        self.id = id
        # taste attributes
        self.genre = random.choice(genre_clusters)
        self.set_style_positioning(all_band_positionings, band_deviation_from_genre)

        if not quietly:
            print("my genre:", self.genre)
            print("my style:", self.style_positioning)
            print("____________________________")

    def set_style_positioning(self, all_band_positionings, band_deviation_from_genre):
        all_band_positionings[0].append(self.id)

        self.style_positioning = []
        i = 1
        for genre_style_point in self.genre:

            trying = True

            while trying:
                deviation = int(np.random.gamma(shape = 2, scale = band_deviation_from_genre)) * random.choice([1, -1])
                if genre_style_point + deviation >=0 and genre_style_point + deviation <= 100:
                    trying = False
                trying = False # comment this line out to enable the 0 <= x <= 100 rule
            
            self.style_positioning.append(
                genre_style_point + deviation
            )

            all_band_positionings[i].append(genre_style_point + deviation)
            i += 1

        self.style_positioning = tuple(self.style_positioning)

    def get_nearest_neighbor(self, all_band_positionings):
        train_df = all_band_positionings[all_band_positionings[0] != self.id] # self is omitted to avoid match with self

        X = train_df.iloc[:, 1:train_df.shape[1]] # features of training set (i.e. columns 1 to last column)
        y = train_df.iloc[:, 0] # first column containing band ids

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X, y)
        prediction = knn.predict([self.style_positioning])

        return prediction[0] # [0] because it returns an array e.g. array([123])

    def prepare_data_for_charting(self, all_band_positionings, genre_clusters):
        df = copy.deepcopy(all_band_positionings)

        df['type'] = 'band'

        genre_clusters_to_chart = copy.deepcopy(genre_clusters)
        for i in range(0, len(genre_clusters_to_chart)):
            genre_clusters_to_chart[i].insert(0, f'genre_{i+1}')

        df_genres = pd.DataFrame(genre_clusters_to_chart)
        df_genres['type'] = "genre"

        df = df.append(df_genres)

        return df

    def plot_nearest_neighbor(self, all_band_positionings, all_bands, genre_clusters):
        df = self.prepare_data_for_charting(all_band_positionings, genre_clusters)

        # Add knn information
        self_position = self.style_positioning
        target_position = all_bands[self.get_nearest_neighbor(all_band_positionings)].style_positioning

        coords_for_line = list(zip(self_position, target_position))

        style_dimensions = df.shape[1]-2

        # Plot data
        if style_dimensions == 2: # 2 dimensions
            fig = px.scatter(df, x=1, y=2, color="type", hover_data=[0])
            fig.add_trace(go.Scatter(x = coords_for_line[0], y = coords_for_line[1]))

        elif style_dimensions == 3: # 3 dimensions
            print('preparing fig')
            fig = px.scatter_3d(df, x=1, y=2, z=3, color="type")
            fig.add_trace(go.Scatter3d(x = coords_for_line[0], y = coords_for_line[1], z = coords_for_line[2]))

        else:
            print(f'Chart only supported for 2 or 3 style dimensions, not {style_dimensions}.')

        fig.show()

    def create_playlist(self, playlist_length, all_band_positionings, all_bands):
        bands_to_consider = copy.deepcopy(all_band_positionings)
        current_band = self

        playlist_bands = [self]
        for i in range(0, playlist_length):
            next_band_id = self.get_nearest_neighbor(bands_to_consider)
            next_band = all_bands[next_band_id]
            playlist_bands.append(next_band)

            bands_to_consider = bands_to_consider[bands_to_consider[0] != current_band.id]

            current_band = all_bands[next_band_id]

        return playlist_bands

    def plot_playlist(self, playlist_length, all_band_positionings, all_bands, genre_clusters):
        playlist_bands = self.create_playlist(playlist_length, all_band_positionings, all_bands)

        playlist_positionings = [band.style_positioning for band in  playlist_bands]

        df = self.prepare_data_for_charting(all_band_positionings, genre_clusters)

        coords_for_line = np.array(playlist_positionings).T.tolist()

        style_dimensions = df.shape[1]-2

        # Plot data
        if style_dimensions == 2: # 2 dimensions
            fig = px.scatter(df, x=1, y=2, color="type", hover_data=[0])
            fig.add_trace(go.Scatter(x = coords_for_line[0], y = coords_for_line[1]))

        elif style_dimensions == 3: # 3 dimensions
            print('preparing fig')
            fig = px.scatter_3d(df, x=1, y=2, z=3, color="type", hover_data=[0])
            fig.add_trace(go.Scatter3d(x = coords_for_line[0], y = coords_for_line[1], z = coords_for_line[2]))

        else:
            print(f'Chart only supported for 2 or 3 style dimensions, not {style_dimensions}.')

        fig.show()