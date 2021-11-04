import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import copy
import plotly.graph_objs as go
import plotly.express as px
import randomname
import band_sim_functions as bsim




class Band:
    def __init__(self, id, genre_clusters, band_deviation_from_genre, all_band_positionings, quietly=True):
        self.id = id
        self.name = randomname.get_name().replace("-", " ").title()
        # taste attributes
        self.genre = random.choice(genre_clusters)
        self.set_style_positioning(all_band_positionings, band_deviation_from_genre)

        if not quietly:
            print("my genre:", self.genre)
            print("my style:", self.style_positioning)
            print("____________________________")

    def set_style_positioning(self, all_band_positionings, band_deviation_from_genre):
        all_band_positionings[0].append(self.id)
        all_band_positionings[1].append(self.name)

        self.style_positioning = []
        i = 2
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
        train_df = all_band_positionings[all_band_positionings["id"] != self.id] # self is omitted to avoid match with self

        X = np.array(train_df.iloc[:, 2:train_df.shape[1]]).tolist() # features of training set (i.e. columns 1 to last column)
        y = np.array(train_df["id"]).tolist() # first column containing band ids

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X, y)
        prediction = knn.predict([self.style_positioning])

        return prediction[0] # [0] because it returns an array e.g. array([123])

    def prepare_data_for_charting(self, all_band_positionings, genre_clusters, style_dimensions):
        df = copy.deepcopy(all_band_positionings)

        df['type'] = 'band'

        genre_clusters_to_chart = copy.deepcopy(genre_clusters)
        for i in range(0, len(genre_clusters_to_chart)):
            genre_clusters_to_chart[i].insert(0, f'genre_{i+1}')
            genre_clusters_to_chart[i].insert(1, f'NA')

        df_genres = pd.DataFrame(genre_clusters_to_chart)
        df_genres = bsim.set_column_names(df_genres, style_dimensions)
        df_genres['type'] = "genre"

        df = df.append(df_genres)

        return df

    def create_playlist(self, playlist_length, all_band_positionings, all_bands):
        bands_to_consider = copy.deepcopy(all_band_positionings)
        bands_to_consider = bands_to_consider[bands_to_consider["id"] != self.id]
        current_band = self

        playlist_bands = [self]
        for i in range(0, playlist_length):
            next_band_id = self.get_nearest_neighbor(bands_to_consider) # change self to current_band for exploration playlist and
            next_band = all_bands[next_band_id]
            playlist_bands.append(next_band)

            bands_to_consider = bands_to_consider[bands_to_consider["id"] != next_band.id] # change next_band to current_band for exploration playlist

            current_band = next_band #all_bands[next_band_id]

        return playlist_bands

    def plot_playlist(self, playlist_length, all_band_positionings, all_bands, genre_clusters, style_dimensions):
        playlist_bands = self.create_playlist(playlist_length, all_band_positionings, all_bands)

        # playlist_positionings = [band.style_positioning for band in playlist_bands]

        df = self.prepare_data_for_charting(all_band_positionings, genre_clusters, style_dimensions)

        # coords_for_line = np.array(playlist_positionings).T.tolist()

        # Plot data
        if style_dimensions == 2: # 2 dimensions
            fig = px.scatter(df, x="x", y="y", color="type", hover_data=["band", "x", "y", "id"], width = 650, height = 650)
            for i in range(0,playlist_length):
                new_trace = np.array((
                    # playlist_bands[i].id,
                    self.style_positioning, 
                    playlist_bands[i].style_positioning
                    # playlist_bands[i].name
                    )).T.tolist()

                fig.add_trace(go.Scatter(x = new_trace[0], y = new_trace[1], line = dict(color="SeaGreen", width = 1)))

            axis_max = max(df["x"].max(), df["y"].max()) + 5
            axis_min = min(df["x"].min(), df["y"].min()) - 5
            fig.update_xaxes(range=[axis_min, axis_max])
            fig.update_yaxes(range=[axis_min, axis_max])


        elif style_dimensions == 3: # 3 dimensions
            print('preparing fig')
            axis_max = max(df["x"].max(), df["y"].max(), df["z"].max()) + 5
            axis_min = min(df["x"].min(), df["y"].min(), df["z"].min()) - 5
            fig = px.scatter_3d(df, x="x", y="y", z="z", color="type", hover_data=["band", "x", "y", "id"], range_x=[axis_min, axis_max], range_y=[axis_min, axis_max], range_z=[axis_min, axis_max])

            for i in range(0,playlist_length):
                new_trace = np.array((self.style_positioning, playlist_bands[i].style_positioning)).T.tolist()
                fig.add_trace(go.Scatter3d(x = new_trace[0], y = new_trace[1], z = new_trace[2], line = dict(color="green", width = 1)))

            fig.update_traces(
                marker=dict(size = 5)
                )

        else:
            print(f'Chart only supported for 2 or 3 style dimensions, not {style_dimensions}.')

        fig.update_layout(showlegend=False)

        fig.show()