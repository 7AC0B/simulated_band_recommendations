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
    df = pd.DataFrame(all_band_positionings).transpose()
    df['type'] = 'band'

    genre_clusters_to_chart = copy.deepcopy(genre_clusters)
    for i in range(0, len(genre_clusters_to_chart)):
        genre_clusters_to_chart[i].insert(0, f'genre_{i+1}')

    df_genres = pd.DataFrame(genre_clusters_to_chart)
    df_genres['type'] = "genre"

    df = df.append(df_genres)

    if print_table:
        display(df)
    
    # Plot data
    if style_dimensions == 2:
        fig = px.scatter(df, x=1, y=2, color="type", hover_data=[0])
    elif style_dimensions == 3:
        fig = px.scatter_3d(df, x=1, y=2, z=3, color="type")
    else:
        print(f'Chart only supported for 2 or 3 style dimensions, not {style_dimensions}.')

    fig.show()

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