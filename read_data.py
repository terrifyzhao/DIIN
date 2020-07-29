import pandas as pd

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('movielens/users.dat', sep='::',
                      header=None, names=unames, engine='python')
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('movielens/ratings.dat', sep='::',
                        header=None, names=rnames, engine='python')
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('movielens/movies.dat', sep='::',
                       header=None, names=mnames, engine='python')

data = pd.merge(pd.merge(ratings, users), movies)
neg = data[data['rating'] <= 1].sample(frac=1)
pos = data[data['rating'] >= 4.5]
pos = pos.sample(frac=1)[0:56000]

neg['label'] = 0
pos['label'] = 1

df = pos.append(neg)
df.to_csv('data.csv', index=False, encoding='utf_8_sig')


