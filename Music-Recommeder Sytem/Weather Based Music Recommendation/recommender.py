from sklearn.preprocessing import StandardScaler

import numpy as np
import requests
import joblib
import json
import psycopg2
import pandas as pd
import copy
import spotipy
from tabulate import tabulate


from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from operator import itemgetter
from spotipy.oauth2 import SpotifyClientCredentials

KEYS = ["danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence"
        ]

MOOD = {
    0: "Unsure",
    1: "Happy",
    2: "Romantic",
    3: "Sad",
    4: "Focused",
    5: "Energetic"
}

# ["Weather", "Energy", "Valence","Less Than =0, Greater Than =1"]
WEATHER_CONDITIONS = {
    'Clear': [0.6, 0, 6, 1],
    'Rain': [0.3, 0.4, 0],
    'Clouds': [0.5, 0.5, 0],
    'Drizzle': [0.5, 0.6, 0],
    'Atmosphere': [0.5, 0.6, 0],
    'Thunderstorm': [0.5, 0.5, 0]
}


def connect_spotify(config):
    """ connect to spotify
    prerequisites: client_id and client_secret"""

    with open(config) as f:
        c = json.load(f)
        client_id = c["client_id"]
        client_secret = c["client_secret"]
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=client_credentials_manager)

def connect_database(config):
    """connecting to database
    prerequisites: database, host, name, port, user, pw"""
    with open(config) as f:
        c = json.load(f)
        conn = psycopg2.connect(
            host=c["db_host"],
            dbname=c["db_name"],
            port=c["db_port"],
            user=c["db_user"],
            password=c["db_pw"])
    return conn

def get_db_data(config):
    """select all data from spotify database"""
    conn = connect_database(config=config)
    cur = conn.cursor()
    cur.execute('SELECT * FROM spotify;')
    return cur.fetchall()


def preprocess_playlist_1(user_file, sp_conn):
    """
    :param sp_conn: spotify connection
    :param user_file: file path to csv file with uris :param sp_conn: authorized spotify connection
    :return:
    - list_of_songs: dictionaries of songs
    - song_matrix: np array with songs data ['danceability', 'energy', 'loudness',
                                            'speechiness', 'acousticness','instrumentalness',
                                            'liveness', 'valence']
    - uri_of_playlist: list of uris
    """

    list_of_songs = []  # store songname
    list_of_uris = []  # store uris for songs
    artist_name = []
    popularity = []
    limit = 100  # maxmum 100 at once
    playlist = pd.read_csv(user_file)[:10]  # todo remove slicing
    l = len(playlist)  # amount of songs
    limitcount = int(l / limit)  # (#songs) /limit
    songleft = 0
    for i in range(l):
        # list_of_uris.append(result['tracks']['items'][0]['uri']) #get the uri of the track(song)
        list_of_uris.append(playlist['uri'][i])
    # To extract the uri by removing 'spotify:track:'
    uri_of_playlist = "".join(list_of_uris)
    uri_of_playlist = uri_of_playlist.split("spotify:track:")
    uri_of_playlist.pop(0)
    # Get all the features of the song
    for i in range(l):
        result = sp_conn.track(uri_of_playlist[i])
        list_of_songs.append([{"Song_name": result["name"]}])
        artist_name.append({"Artist": result["artists"][0]["name"]})  # get the artist_name
        popularity.append({"popularity": result['popularity']})  # get the popularity of the song
    for j in range(limitcount + 2):
        songleft = l - j * limit
        if songleft > limit:
            Features = sp_conn.audio_features(uri_of_playlist[j * limit:j * limit + limit])
            for i in range(limit):
                list_of_songs[j * limit + i][0].update(artist_name[i])
                list_of_songs[j * limit + i][0].update(popularity[i])
                list_of_songs[j * limit + i][0].update(Features[i])
                del list_of_songs[j * limit + i][0]['type']
                del list_of_songs[j * limit + i][0]['id']
                del list_of_songs[j * limit + i][0]['uri']
                del list_of_songs[j * limit + i][0]['track_href']
                del list_of_songs[j * limit + i][0]['analysis_url']
        elif (songleft <= limit and songleft > 0):
            Features = sp_conn.audio_features(uri_of_playlist[j * limit:j * limit + songleft])
            for i in range(songleft):
                list_of_songs[j * limit + i][0].update(artist_name[i])
                list_of_songs[j * limit + i][0].update(popularity[i])
                list_of_songs[j * limit + i][0].update(Features[i])
                del list_of_songs[j * limit + i][0]['type']
                del list_of_songs[j * limit + i][0]['id']
                del list_of_songs[j * limit + i][0]['uri']
                del list_of_songs[j * limit + i][0]['track_href']
                del list_of_songs[j * limit + i][0]['analysis_url']
        else:
            # convert playlist to numpy array
            song_matrix = []
            l = len(list_of_songs)
            for j in range(l):
                arr = []
                for i in range(8):
                    arr.append(list_of_songs[j][0][KEYS[i]])
                song_matrix.append(np.array(arr))

            return list_of_songs, song_matrix, uri_of_playlist


def db_to_normal_array(db_res):
    """convert databse to numpy arrary withou preprocessing"""
    l = len(db_res)
    train_X = []
    for i in range(l):
        song_matrix = np.array(list(db_res[i][1:9]))
        train_X.append(song_matrix)
    return train_X


def get_std_playlist(features_arr, uri_arr, db_cont):
    """get standardscaler processed user playlist"""
    db_tmp = db_to_normal_array(db_cont)
    index = []
    flag = 0
    for i in range(len(uri_arr)):
        for item in db_cont:

            if uri_arr[i] == item[0]:
                flag = 1
                index.append(db_cont.index(item))
                break
        if flag == 0:  # not found in db
            db_tmp.append(features_arr[i])
            index.append(len(db_tmp) - 1)
        if flag == 1:
            flag = 0

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(db_tmp)
    test_X = []
    for item in index:
        test_X.append(scaled_features[item])
    test_X = np.array(test_X)

    return test_X


def predict_moods_db(model, db_data):
    db_tmp = db_to_normal_array(db_data)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(db_tmp)
    test_y = []
    # Feature selection : select feature from 3 to 5  which are speechiness,acousticness,
    # and instrumentalness
    for i in range(len(scaled_features)):
        test_y.append(scaled_features[i][3:6])

    y_pred = model.predict(test_y)  # mods for every entry in spotify table

    db_moods = []
    for i, data in enumerate(db_data):
        new_d = list(data)
        new_d.append(y_pred[i])
        db_moods.append(new_d)
    return db_moods


##################### KNN #############################
def preprocess_playlist(db_data, playlist, sp_conn):
    """
    :param sp_conn: spotify connection
    :param user_file: file path to csv file with uris :param sp_conn: authorized spotify connection
    :return:
    - list_of_songs: dictionaries of songs
    - song_matrix: np array with songs data ['danceability', 'energy', 'loudness',
                                            'speechiness', 'acousticness','instrumentalness',
                                            'liveness', 'valence']
    - uri_of_playlist: list of uris
    """

    pre_proc_songs = []

    # convert to 2d list and keep only id
    playlist = playlist.to_numpy().tolist()
    playlist_uris = [x[0].split('spotify:track:')[1] for x in playlist]  # todo THIS NEEDS SLICE

    uris_found_in_db = []
    # if song is in database add it
    for uri in playlist_uris:
        for song in db_data:
            if uri in song:
                pre_proc_songs.append(song)
                uris_found_in_db.append(uri)
                break

    # else extract data from spotipy + add to database
    uris_not_in_db = list(set(playlist_uris) - set(uris_found_in_db))

    limit = 100  # maxmum 100 at once
    l = len(uris_not_in_db)  # amount of songs
    limitcount = int(l / limit)  # (#songs) /limit

    for j in range(limitcount + 2):
        songleft = l - j * limit
        if songleft > limit:
            features = sp_conn.audio_features(uris_not_in_db[j * limit:j * limit + limit])
            for feat in features:
                result = sp_conn.track(feat['id'])
                song = [feat['id'], feat['danceability'], feat['energy'],
                        feat['loudness'], feat['speechiness'], feat['acousticness'],
                        feat['instrumentalness'], feat['liveness'], feat['valence'],
                        result['popularity'], 'None']  # no genre :/
                pre_proc_songs.append(song)

        elif (songleft <= limit and songleft > 0):
            features = sp_conn.audio_features(uris_not_in_db[j * limit:j * limit + songleft])
            for feat in features:
                result = sp_conn.track(feat['id'])
                song = [feat['id'], feat['danceability'], feat['energy'],
                        feat['loudness'], feat['speechiness'], feat['acousticness'],
                        feat['instrumentalness'], feat['liveness'], feat['valence'],
                        result['popularity'], 'None']  # no genre :/
                pre_proc_songs.append(song)

        else:
            return pre_proc_songs


def preprocess_data_playlist(raw_data, user_playlist_file, spotify_conn):
    """@raw_data: list of songs with their features from db
        @user_playlist: csv with uris
        @return: pre-processed set of songs incl user_song"""

    playlist = pd.read_csv(user_playlist_file)
    # pick random 20% of songs
    playlist = playlist.sample(frac=0.2)

    return preprocess_playlist(raw_data, playlist, spotify_conn)


# KNN prediction for playlist
def preprocess_playlist(db_data, playlist, sp_conn):
    """
    :param sp_conn: spotify connection
    :param user_file: file path to csv file with uris :param sp_conn: authorized spotify connection
    :return:
    - list_of_songs: dictionaries of songs
    - song_matrix: np array with songs data ['danceability', 'energy', 'loudness',
                                            'speechiness', 'acousticness','instrumentalness',
                                            'liveness', 'valence']
    - uri_of_playlist: list of uris
    """

    pre_proc_songs = []

    # convert to 2d list and keep only id
    playlist = playlist.to_numpy().tolist()
    playlist_uris = [x[0].split('spotify:track:')[1] for x in playlist][:10]  # todo THIS NEEDS SLICE

    uris_found_in_db = []
    # if song is in database add it
    for uri in playlist_uris:
        for song in db_data:
            if uri in song:
                pre_proc_songs.append(song)
                uris_found_in_db.append(uri)
                break

    # else extract data from spotipy + add to database
    uris_not_in_db = list(set(playlist_uris) - set(uris_found_in_db))

    limit = 100  # maxmum 100 at once
    l = len(uris_not_in_db)  # amount of songs
    limitcount = int(l / limit)  # (#songs) /limit

    for j in range(limitcount + 2):
        songleft = l - j * limit
        if songleft > limit:
            features = sp_conn.audio_features(uris_not_in_db[j * limit:j * limit + limit])
            for feat in features:
                result = sp_conn.track(feat['id'])
                song = [feat['id'], feat['danceability'], feat['energy'],
                        feat['loudness'], feat['speechiness'], feat['acousticness'],
                        feat['instrumentalness'], feat['liveness'], feat['valence'],
                        result['popularity'], 'None']  # no genre :/
                pre_proc_songs.append(song)

        elif (songleft <= limit and songleft > 0):
            features = sp_conn.audio_features(uris_not_in_db[j * limit:j * limit + songleft])
            for feat in features:
                result = sp_conn.track(feat['id'])
                song = [feat['id'], feat['danceability'], feat['energy'],
                        feat['loudness'], feat['speechiness'], feat['acousticness'],
                        feat['instrumentalness'], feat['liveness'], feat['valence'],
                        result['popularity'], 'None']  # no genre :/
                pre_proc_songs.append(song)

        else:
            return pre_proc_songs


def preprocess_data_playlist(raw_data, user_playlist_file, spotify_conn, rf_model, kn_model):
    """@raw_data: list of songs with their features from db
      @user_playlist: csv with uris
      @return: pre-processed set of songs incl user_song"""

    playlist = pd.read_csv(user_playlist_file)
    # pick random 20% of songs
    playlist = playlist.sample(frac=0.2)

    # u_songs, u_uris = preprocess_playlist(raw_data, user_playlist_file, spotify_conn)
    pre_playlist = preprocess_playlist(raw_data, playlist, spotify_conn)
    # remove genre for KNN in playlist and db since some genre is None
    playlist = [song[:-1] for song in pre_playlist]
    raw_data_no_genre = [song[:-1] for song in raw_data]
    # predict mood for database and users_playlist
    # rf_model = joblib.load(saved_modelRF)
    playlist_mood = predict_moods_db(rf_model, playlist)
    db_mood = predict_moods_db(kn_model, raw_data_no_genre)
    db_FULL = predict_moods_db(kn_model, raw_data)
    LEN_PLAYLIST = len(playlist)

    # append user's song to the end for later indexing
    for user_song in playlist_mood:
        if user_song in db_mood:
            db_mood.remove(user_song)
        db_mood.append(user_song)

    # get rid of id (since it is string)
    indices = list(range(1, len(db_mood[0])))
    data_no_uri = [itemgetter(*indices)(entry) for entry in db_mood]
    # normalize everything except mood
    norm_data = normalize(data_no_uri).tolist()
    pre_data = []
    for n_song, c_song in zip(norm_data, data_no_uri):
        temp_data = copy.deepcopy(n_song[:-1])
        temp_data.append(c_song[-1])
        pre_data.append(temp_data)

    # apply weights (make sure the last value is 1! Because otherwise it will change mood)
    # danceability | energy | loudness | speechiness | acousticness | instrumentalness
    # | liveness | valence | popularity | genre
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 0.3, 1]
    weighted_norm_data = []
    for song_entry in pre_data:
        new_song = []
        for i, value in enumerate(song_entry):
            value = value * weights[i]
            new_song.append(value)
        weighted_norm_data.append(new_song)

    weighted_norm_data = [tuple(weighted_entry) for weighted_entry in weighted_norm_data]
    return weighted_norm_data, len(playlist_mood), db_FULL


def get_recomendation_KNN(cooked_data, num_of_user_songs):
    """recommend top 2 nearest songs to every users song in playlist (last n songs in cooked_data)
        prerequisite: data is normalized and weighted (cooked)"""

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(cooked_data)
    distances, indices = nbrs.kneighbors(cooked_data)
    distances, indices = distances.tolist(), indices.tolist()
    # find n-cluster with the user song
    # user songs were appended last, so its indices are:
    user_songs_i = list(range(len(cooked_data) - num_of_user_songs, len(cooked_data)))

    user_songs_neigbors = []  # first element of list = index of user_song (id of song itself), value = list of distances and list of neigbors for that song

    for user_song_i in user_songs_i:
        single_song_neighbors = []
        for ind_list in zip(distances, indices):
            if user_song_i in ind_list[1]:
                single_song_neighbors.append(list(ind_list))
        full = [user_song_i]
        full.append(single_song_neighbors)
        user_songs_neigbors.append(full)

    # select the best cluster -> smallest sum of distances
    # all_songs_best_5 = {} # index of song, best 5 songs indexes as list, # here is sort them by their original song from that the recommendations come
    all_best_songs = []  # here is just a bunch of recommendation independent from original song/ removed users song
    user_songs = []
    for neighbors in user_songs_neigbors:
        song_index = neighbors[0]
        sums = [sum(dist[0]) for dist in neighbors[1]]
        best_5 = neighbors[1][sums.index(min(sums))]
        best_5_i = [ind for ind in best_5[1] if ind != song_index]
        user_songs.append(song_index)
        all_best_songs.append(best_5_i)

    return user_songs, all_best_songs


def search_db(list_idnx, db_data):
    songs_out = []
    for recommendations in list_idnx:
        for indx in recommendations:
            songs_out.append(db_data[indx])
    return songs_out


def filter_by_weather(config, data):
    with open(config) as f:
        config = json.load(f)

    ip_key = config['ip_key']
    weather_key = config['weather_key']

    # get current location
    url = f'http://api.ipstack.com/check?access_key={ip_key}'
    geo_req = json.loads(requests.get(url).text)
    lat = geo_req['latitude']
    lon = geo_req['longitude']

    # get weather of location
    url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weather_key}'
    weather_condition = json.loads(requests.get(url).text)['weather'][0]['main']
    filter_cond = WEATHER_CONDITIONS[weather_condition]
    energy, valence, m_l = filter_cond[0], filter_cond[1], filter_cond[2]

    fltrd = []
    for entry in data:
        if m_l == 0:
            if (entry[2] < energy) and (entry[8] < valence):
                fltrd.append(entry)
        if m_l == 1:
            if (entry[2] > energy) and (entry[8] > valence):
                fltrd.append(entry)

    return weather_condition, fltrd


def visualize(data):
    recommendations_result = []
    for recomendated_data in data:
        res = sp.track(recomendated_data[0])
        artist = res['artists'][0]['name']
        song = res['name']
        recommendations_result.append([recomendated_data[0], artist, song,
                                       recomendated_data[-3], recomendated_data[-2], MOOD[recomendated_data[-1]]])

    df = pd.DataFrame(recommendations_result, columns=['URI', 'Artist', 'Song', 'Popularity', 'Genre', 'Mood'])
    df.set_index('URI', inplace=True)
    df.sort_values(by=['Popularity'], inplace=True, ascending=False)
    return df


# MAIN
if __name__ == '__main__':
    # Prerequisite: config files
    # from google.colab import drive

    # drive.mount('/content/gdrive')

    # root = '/content/gdrive/My Drive/code/'
    root = ''
    config = root + 'config.json'
    nishant = root + 'Nishant2500.csv'

    sp = connect_spotify(config)
    db_data = cur = get_db_data(config)

    saved_modelRF = root + 'RFmodel.pkl'
    km_model_path = root + 'Kstd6.pkl'

    # load models
    km_model = joblib.load(km_model_path)
    rf_model = joblib.load(saved_modelRF)

    # preprocess data format and get the audio features, normalization and weights.
    # db_full is with genre and mood
    data_playlist_pre_processed, num_user_songs, db_full = preprocess_data_playlist(db_data, nishant, sp, rf_model,
                                                                                    km_model)
    # recommend with the preprocessed playlist data
    given_songs, recommendations = get_recomendation_KNN(data_playlist_pre_processed, num_user_songs)

    # find recommended songs in database by their index
    result_data = search_db(recommendations, db_full)
    # filter recommendations by weather conditions
    weather, result_fltrd = filter_by_weather(config, result_data)

    # visualize results
    print(f'These are the songs recommendations for you. BTW, it is {weather} outiside!')
    table = visualize(result_fltrd)
    print(tabulate(table, headers='keys', tablefmt='pretty'))