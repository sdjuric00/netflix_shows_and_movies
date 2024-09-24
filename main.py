import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.preprocessing import MinMaxScaler


# Kreiramo skalir
scaler = MinMaxScaler()


def basic_analysis(titles_data, credits_data):
    # Provera prvih 5 redova
    print(f"\nTitles:\n {titles_data.head()}")
    print(f"\nCredits:\n {credits_data.head()}")

    #provera osnovnih informacija
    print(f"\nTitles info:\n {titles_data.info()}")
    print(f"\nCredits info:\n {credits_data.info()}")

    # Podešavanje float formata za prikaz decimalnih brojeva
    pd.options.display.float_format = '{:.4f}'.format

    # Podesavanje opcija prikaza za sve kolone
    pd.set_option('display.max_columns', None)

    # Osnovne statistike skupa
    print(f"\nOsnovna statistika Titles:\n {titles_data.describe()}")
    print(f"\nOsnovna statistika Credits:\n {credits_data.describe()}")

    # Prikaz kolona koje imaju nedostajuće vrednosti
    missing_columns_titles = titles_data.columns[titles_data.isnull().any()]
    missing_columns_credits = credits_data.columns[credits_data.isnull().any()]
    print(f"\nKolone sa nedostajucim vrednostima u Titles: {missing_columns_titles}")
    print(f"\nKolone sa nedostajucim vrednostima u Credits: {missing_columns_credits}")

    # Procenat nedostajućih vrednosti po kolonama u Titles
    missing_percentage = titles_data.isnull().mean() * 100
    print(f"\nProcenat nedostajućih vrednosti po kolonama u Titles:\n {missing_percentage}")
    print(f"\nProcenat nedostajućih vrednosti po kolonama u Credits:\n {credits_data.isnull().mean() * 100}")

    # HeatMap za nedostajuce vrednosti
    plt.subplots(figsize=(16, 14))
    sns.heatmap(titles_data.isnull(), cbar=False, cmap='Reds')
    plt.show()

    print("\nMoguce sezone za serije:\n", np.sort(titles_data['seasons'].unique()))
    print("\nBroj jedinstvenih sezona:\n", titles_data['seasons'].unique().size)


def correlation_matrix_ratings(titles_data):
    # Selekcija relevantnih obeležja
    relevant_columns = ['imdb_score', 'tmdb_score', 'imdb_votes_normalized', 'tmdb_popularity_normalized']

    # Normalizacija imdb_votes i tmdb_popularity
    titles_data[['imdb_votes_normalized', 'tmdb_popularity_normalized']] = scaler.fit_transform(
        titles_data[['imdb_votes', 'tmdb_popularity']]
    )

    # Korelaciona matrica
    correlation_matrix = titles_data[relevant_columns].corr()

    # Vizualizacija korelacionih odnosa pomoću heatmap-a
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Korelaciona matrica između IMDb i TMDB ocena, glasova i popularnosti')
    plt.show()


def data_cleaning(titles_data):
    #uklanjanje outliera
    cleaned_data = titles_data.drop(titles_data[titles_data['runtime'] == 0].index)

    #odbacivanje age_classification i IMDb votes
    cleaned_data = cleaned_data.drop(columns=['age_certification'])
    cleaned_data = cleaned_data.drop(columns=['imdb_votes'])

    # Popunjavanje nedostajućih vrednosti median tehnikom
    cleaned_data['imdb_score'].fillna(cleaned_data['imdb_score'].median(), inplace=True)
    cleaned_data['tmdb_popularity'].fillna(cleaned_data['tmdb_popularity'].median(), inplace=True)
    cleaned_data['tmdb_score'].fillna(cleaned_data['tmdb_score'].median(), inplace=True)

    return cleaned_data


def combine_IMDb_TMDB_ratings(titles_data):
    # Kombinovanje IMDb ocena i TMDB ocena
    titles_data['combined_rating'] = ((titles_data['imdb_score'] * 0.6) + (titles_data['tmdb_score'] * 0.4))

    # Prikaz prvih nekoliko redova sa novim obeležjem
    print(titles_data[['title', 'combined_rating']].head())

    return titles_data


def analyse_score_by_production_country(titles_data):
    # Funkcija za pretvaranje stringova u liste (ast.literal_eval)
    titles_data['production_countries'] = titles_data['production_countries'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # Uklanjanje redova gde je production_countries prazna lista
    titles_data = titles_data[titles_data['production_countries'].apply(lambda x: len(x) > 0)]

    # Podela uzorka na filmove i serije
    movies_data = titles_data[titles_data['type'] == 'MOVIE']
    series_data = titles_data[titles_data['type'] == 'SHOW']

    # Dobavljanje top 15 zemalja po proizvodnji filmova i serija (koristeći explode da proširimo liste zemalja)
    movies_top_countries = movies_data.explode('production_countries')['production_countries'].value_counts().head(15)
    series_top_countries = series_data.explode('production_countries')['production_countries'].value_counts().head(15)

    # Filtriranje filmova i serija samo za top 15 zemalja
    movies_data_top_15 = movies_data[movies_data['production_countries'].apply(
        lambda x: any(country in movies_top_countries.index for country in x))]
    series_data_top_15 = series_data[series_data['production_countries'].apply(
        lambda x: any(country in series_top_countries.index for country in x))]

    # Vizualizacija za filmove - Top 15 zemalja po broju proizvedenih filmova
    plt.figure(figsize=(10, 6))
    sns.barplot(x=movies_top_countries.values, y=movies_top_countries.index, color='blue')
    plt.title("Top 15 zemalja po proizvodnji filmova")
    plt.xlabel("Broj filmova")
    plt.ylabel("Država")
    plt.show()

    # Vizualizacija za serije - Top 15 zemalja po broju proizvedenih serija
    plt.figure(figsize=(10, 6))
    sns.barplot(x=series_top_countries.values, y=series_top_countries.index, color='green')
    plt.title("Top 15 zemalja po proizvodnji serija")
    plt.xlabel("Broj serija")
    plt.ylabel("Država")
    plt.show()

    # Računanje prosečnih ocena po državama za filmove
    movies_avg_ratings = movies_data_top_15.explode('production_countries').groupby('production_countries').agg({
        'combined_rating': 'mean'
    }).rename(columns={'combined_rating': 'average_score'}).loc[movies_top_countries.index]

    # Vizualizacija prosečnih ocena po državama za filmove
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=movies_avg_ratings['average_score'], y=movies_avg_ratings.index, color='blue')
    sns.barplot(x=movies_avg_ratings['average_score'], y=movies_avg_ratings.index, color='blue')
    plt.title("Prosečne ocene filmova po državama (Top 15 zemalja)")
    plt.xlabel("Prosečna ocena")
    plt.ylabel("Država")
    for bar in bars.patches:
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.1f}', ha='center', va='center')

    plt.show()

    # Računanje prosečnih ocena po državama za serije
    series_avg_ratings = series_data_top_15.explode('production_countries').groupby('production_countries').agg({
        'combined_rating': 'mean'
    }).rename(columns={'combined_rating': 'average_score'}).loc[series_top_countries.index]

    # Vizualizacija prosečnih ocena po državama za serije
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=series_avg_ratings['average_score'], y=series_avg_ratings.index, color='green')
    sns.barplot(x=series_avg_ratings['average_score'], y=series_avg_ratings.index, color='green')
    plt.title("Prosečne ocene serija po državama (Top 15 zemalja)")
    plt.xlabel("Prosečna ocena")
    plt.ylabel("Država")
    for bar in bars.patches:
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.1f}', ha='center', va='center')

    plt.show()


def analyse_score_by_content_type(titles_data):
    # Poređenje prosečnih ocena između filmova i serija
    movies = titles_data[titles_data['type'] == 'MOVIE']
    shows = titles_data[titles_data['type'] == 'SHOW']

    movies_avg_score = movies['combined_rating'].mean()
    series_avg_score = shows['combined_rating'].mean()

    print(f"Prosečna ocena filmova: {movies_avg_score}")
    print(f"Prosečna ocena serija: {series_avg_score}")

    sns.histplot(data=movies, x='imdb_score', label='MOVIE', alpha=0.5)
    sns.histplot(data=shows, x='imdb_score', label='SHOW', alpha=0.5)

    # Set labels and legend
    plt.xlabel('Ocena')
    plt.ylabel('Broj instanci sa specificnom ocenom')
    plt.legend()

    #Odnos filmova i serija
    content_types = titles_data.groupby('type').size().reset_index().rename(columns={0: 'counts'})

    langs = content_types.type
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    plt.tight_layout(pad=2)
    sns.set_style("darkgrid")
    a1 = sns.barplot(x=content_types.type, y=content_types.counts, ax=axes[0], palette='coolwarm');
    a1.set(xlabel='Types', ylabel='Counts')
    plt.pie(content_types.counts, autopct='%1.2f%%', labels=langs, radius=1.5, labeldistance=1.1, rotatelabels=True)
    plt.legend()
    plt.show()

    #prosecna ocena u odnosu na broj sezona serije

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Korelacija između broja sezona i ocene
    sns.regplot(x='seasons', y='combined_rating', data=titles_data, ax=axes[0], scatter_kws={'color': 'blue'},
                line_kws={'color': 'red'})
    axes[0].set_title('Broj sezona vs Prosečna ocena')
    axes[0].set_xlabel('Broj sezona')
    axes[0].set_ylabel('Prosečna ocena')

    # Korelacija između broja sezona i TMDB popularnosti
    sns.regplot(x='seasons', y='tmdb_popularity', data=titles_data, ax=axes[1], scatter_kws={'color': 'green'},
                line_kws={'color': 'orange'})
    axes[1].set_title('Broj sezona vs TMDB popularnost')
    axes[1].set_xlabel('Broj sezona')
    axes[1].set_ylabel('TMDB popularnost')
    plt.tight_layout()
    plt.show()

    # Group series by number of seasons to see which range has the highest average rating
    avg_rating_per_seasons = titles_data[titles_data['type'] == 'SHOW'].groupby('seasons')['combined_rating'].mean()

    #prosecna ocena u odnosu na duzinu trajanja filma

    # Analiza dužine trajanja filmova i ocena
    plt.figure(figsize=(10, 6))

    # Scatter plot sa linijom regresije
    sns.regplot(x='runtime', y='combined_rating', data=titles_data[titles_data['type'] == 'MOVIE'],
                scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})

    # Naslovi i oznake
    plt.title("Odnos između dužine trajanja filma i ocene")
    plt.xlabel("Dužina trajanja (minuti)")
    plt.ylabel("Prosečna ocena")

    # Prikaz grafa
    plt.show()


def analyse_score_by_genre(titles_data):
    # Funkcija za pretvaranje stringova u liste
    titles_data['genres'] = titles_data['genres'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # Uklanjanje redova gde je genres prazna lista
    titles_data = titles_data[titles_data['genres'].apply(lambda x: len(x) > 0)]

    # Eksplodiranje žanrova da bismo dobili jedinstvene žanrove
    titles_data_exploded = titles_data.explode('genres')

    # Računanje prosečnih ocena i popularnosti po žanrovima
    genres_ratings = titles_data_exploded.groupby("genres").aggregate(
        {"combined_rating": "mean", "tmdb_popularity": "mean"})

    # Uzmi najpopularnijih 10 žanrova prema prosečnoj oceni i popularnosti
    top_genres_rating = genres_ratings.nlargest(10, 'combined_rating')
    top_genres_popularity = genres_ratings.nlargest(10, 'tmdb_popularity')

    # Kreiranje barplota, rating
    plt.figure(figsize=(18, 9))
    sns.barplot(x=top_genres_rating.index, y=top_genres_rating['combined_rating']).set(
        xlabel='Žanr',
        ylabel='Ocena',
        title='Top 10 Žanrova prema Oceni'
    )
    plt.xticks(rotation=45)
    plt.show()

    # Kreiranje barplota, TMDB popularity
    plt.figure(figsize=(18, 9))
    sns.barplot(x=top_genres_popularity.index, y=top_genres_popularity['tmdb_popularity']).set(
        xlabel='Žanr',
        ylabel='TMDB Popularity',
        title='Top 10 Žanrova prema popularnosti'
    )
    plt.xticks(rotation=45)
    plt.show()


def analyse_score_by_credits(titles_data, credits_data):
    # Odnos reditelja i glumaca
    sns.countplot(x='role', data=credits_data)
    plt.title('Broj glumaca u odnosu na reditelje')
    plt.show()

    merged_datasets = pd.merge(titles_data, credits_data, on='id')
    merged_datasets = merged_datasets[merged_datasets['release_year'] >= 2015] #uzimaju se u obzir samo naslovi noviji od 2015
    most_popular_actors_directors = merged_datasets.groupby('name').agg({'id': 'count'}).reset_index()

    most_popular_actors_directors = most_popular_actors_directors.sort_values(by='id', ascending=False)

    # Najbolje ocenjeni glumci koji su glumili u barem 3 naslova
    top_actors = merged_datasets[merged_datasets['role'] == 'ACTOR'].groupby('name').filter(lambda x: len(x) >= 6)
    avg_scores_actors = top_actors.groupby('name')['combined_rating'].mean().reset_index()

    top_10_score_actors = avg_scores_actors.sort_values(by='combined_rating', ascending=False).head(10)

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='name', y='combined_rating', data=top_10_score_actors)
    plt.title('Prosečna ocena za najbolje ocenjene glumce', fontsize=16)
    plt.xlabel('Glumac', fontsize=14)
    plt.ylabel('Prosečna ocena', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Najbolje ocenjeni reditelji koji su režirali bar 3 naslova
    top_directors = merged_datasets[merged_datasets['role'] == 'DIRECTOR'].groupby('name').filter(lambda x: len(x) >= 6)
    avg_scores_directors = top_directors.groupby('name')['combined_rating'].mean().reset_index()

    top_10_score_directors = avg_scores_directors.sort_values(by='combined_rating', ascending=False).head(10)

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    sns.boxplot(x='name', y='combined_rating', data=top_10_score_directors)
    plt.title('Prosečna ocena za najbolje ocenjene reditelje', fontsize=16)
    plt.xlabel('Reditelj', fontsize=14)
    plt.ylabel('Prosečna ocena', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Učitavanje podataka iz dataseta
    titles_data = pd.read_csv('data/titles.csv')
    credits_data = pd.read_csv('data/credits.csv')

    basic_analysis(titles_data, credits_data)
    correlation_matrix_ratings(titles_data)

    titles_data = data_cleaning(titles_data)
    titles_data = combine_IMDb_TMDB_ratings(titles_data)

    analyse_score_by_production_country(titles_data)
    analyse_score_by_content_type(titles_data)
    analyse_score_by_genre(titles_data)
    analyse_score_by_credits(titles_data, credits_data)


