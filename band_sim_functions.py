import random
import copy
import pandas as pd
import plotly.express as px

def generate_random_cluster_locations(dimensions = 3, cluster_count = 4):
    cluster_locations = []
    for c in range(0,cluster_count):

        coordinate = []
        for d in range(0, dimensions):
            coordinate.append(
                random.randint(0,100)
            )
        
        cluster_locations.append(
            (coordinate)
        )

    return cluster_locations


def plot_all_band_positions(all_band_positionings, genre_clusters, style_dimensions, print_table = False):
    # Prepare data
    df = all_band_positionings
    df['type'] = 'band'

    genre_clusters_to_chart = copy.deepcopy(genre_clusters)
    for i in range(0, len(genre_clusters_to_chart)):
        genre_clusters_to_chart[i].insert(0, f'genre_{i+1}')
        genre_clusters_to_chart[i].insert(1, f'NA')


    df_genres = pd.DataFrame(genre_clusters_to_chart)
    df_genres = set_column_names(df_genres, style_dimensions)
    df_genres['type'] = "genre"

    df = df.append(df_genres)

    if print_table:
        display(df)
    
    # Plot data
    if style_dimensions == 2:
        fig = px.scatter(df, x="x", y="y", color="type", hover_data=["band", "x", "y", "id"])
    elif style_dimensions == 3:
        fig = px.scatter_3d(df, x="x", y="y", z="z", hover_data=["band", "x", "y", "z", "id"], color="type")
    else:
        print(f'Chart only supported for 2 or 3 style dimensions, not {style_dimensions}.')

    fig.update_traces(
        marker=dict(size = 5)
    )

    fig.show()

def set_column_names(df, style_dimensions):
    if style_dimensions == 2:
        df.columns = ["id", "band", "x", "y"]
    elif style_dimensions == 3:
        df.columns = ["id", "band", "x", "y", "z"]
    else:
        pass
    
    return df

#### HELPERS ####

def create_LoL(dims = 3):
    LoL = []
    for i in range(0,dims):
        LoL.append([])
    return(LoL)



#### RESOURCES ####
'''
https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea
https://towardsdatascience.com/tagged/real-world-examples-knn?p=23832490e3f4

'''