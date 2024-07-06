import random, math
import numpy as np
import json, os, pickle
import pandas as pd
import sqlite3
from partition import validate

# from eval import *

"""
Data Module: randomly generate paper and author data
Here are the data structures that store paper and author information:
papers = [
    {
        "true_score" : float,
        "rev_score" : float,
    },
    ...
]
authors = [
    {
        paper_idx : score,
        ...
    },
    ...
]
"""

class Conference:
    def __init__(self, papers, authors):
        self.papers = papers
        self.authors = authors
        self.m = len(authors) 
        self.n = len(papers)
        self.graph = self.generate_graph()

    def generate_graph(self):
        graph = {}
        for i, author in enumerate(self.authors):
            graph[i] = set( int(k) for k in author.keys())
        return graph
    # def save(self, filename):
    # 	with open(filename, 'wb') as f:
    # 		pickle.dump(self, f)




papers_file_name = 'papers.json'
authors_file_name = 'authors.json'

def save_data(papers, authors, data_dir):
    with open(data_dir + papers_file_name, 'w') as outfile:
        json.dump(papers, outfile, indent=4, cls=NpEncoder)
    with open(data_dir + authors_file_name, 'w') as outfile:
        json.dump(authors, outfile, indent=4, cls=NpEncoder)

def load_data(data_dir):
    with open(data_dir + papers_file_name, 'r') as outfile:
        papers = json.load(outfile)
    with open(data_dir + authors_file_name, 'r') as outfile:
        authors = json.load(outfile)
    return papers, authors

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_conference(conference, data_dir, name):
    conference = Conference(papers, authors)
    with open(data_dir + name + '.pickle', 'wb') as f:
        pickle.dump(conference, f)

def load_conference(data_dir, name):
    with open(data_dir + name + '.pickle', 'rb') as f:
        conference = pickle.load(f)
    return conference


def sample_score(i, review_noise):
    ### iclr simulation
    true_score = random.random()
    review_score = true_score + random.random() * review_noise
    return true_score, review_score



def generate_from_graph(graph, m, n, review_noise=0.5, self_noise=0, score_func=sample_score):
    papers = []
    authors = []

    for i in range(n):
        true_score, review_score = score_func(i, review_noise)
        # if math.isnan(true_score) or math.isnan(review_score):
        #     print(i, true_score, review_score)
        papers.append({
            "true_score" : true_score,
            "rev_score" : review_score 
        })

    for i in range(m):
        author = dict()
        for j in graph[i]:
            author[j] = papers[j]['true_score'] + random.random() * self_noise
        authors.append(author)

    return papers, authors

def gen_from_iclr(conference_name):
    
    if conference_name == 'iclr21': ### iclr 21
        data_dir = 'data/' + conference_name + '/'
        csv_file = open(data_dir + 'authorlist.csv', 'r')
        df = pd.read_csv(csv_file)   
        review_file = open(data_dir + 'ratings.tsv', 'r')
        review_df = pd.read_csv(review_file, sep='\t')
        score_col = '0'
    elif conference_name == 'iclr22': ### iclr 22
        score_col = 'rating_0_avg' ## 'avg'
        data_dir = 'data/' + conference_name + '/'
        db_name = conference_name + '.db'
        cmd = 'SELECT url, rating_0_avg, rating_3_avg FROM submissions;'

        csv_file = open(data_dir + 'authorlist.csv', 'r')
        df = pd.read_csv(csv_file)   
        database = sqlite3.connect(data_dir + db_name, check_same_thread=False)
        review_df = pd.read_sql(cmd, database)

        review_df['paper_id'] = review_df['url'].apply(lambda x: x.split('=')[-1])
    elif conference_name == 'iclr23': ### iclr 23
        score_col = 's_0_avg' ## 'avg'
        data_dir = 'data/' + conference_name + '/'
        db_name = conference_name + '.db'
        cmd = 'SELECT url_id, s_0_avg, s_1_avg FROM submissions;'

        csv_file = open(data_dir + 'authorlist.csv', 'r')
        df = pd.read_csv(csv_file)   
        database = sqlite3.connect(data_dir + db_name, check_same_thread=False)
        review_df = pd.read_sql(cmd, database)

        ## iclr 23
        review_df['paper_id'] = review_df['url_id']


    df = df[df['paper_id'].isin(review_df['paper_id'])]
    df = df.merge(review_df, on='paper_id')
    df =  df[  df[score_col].isnull() == False ]

    authors = df['author_id'].unique().tolist()
    papers = df['paper_id'].unique().tolist()
    paperid2idx = {papers[i]: i for i in range(len(papers))}
    
    idx2rating = [df[ df['paper_id'] == papers[i] ][score_col].values[0] for i in range(len(papers))]
    m = len(authors)
    n = len(papers)

    print(m, n)

    graph = {}
    for i, author in enumerate(authors):
        graph[i] = set()
        for paper in df[df['author_id'] == author]['paper_id']:
            graph[i].add(paperid2idx[paper])

    scoring = lambda x, noise: (idx2rating[x] + random.gauss(0, noise), idx2rating[x])

    for review_noise in [1, 2, 3, 4]:
        for self_noise in [0, 0.1, 0.5, 1, 2]:
            for instance in range(5):
                papers, authors = generate_from_graph(graph, m, n, self_noise=self_noise, review_noise=review_noise, score_func=scoring)
                conference = Conference(papers, authors)
                save_conference(conference, data_dir, '{}-{}-{}-{}'.format(conference_name, review_noise, self_noise, instance))

def tree_graph(depth, base=2, nary=2): 
    num_papers = 0
    num_authors = 0
    graph = {}
    ### authors at height = d owns papers for all its child at height = d - 1
    
    for i in range(nary ** depth): ### authors at height = 0
        paper_set = set()
        for j in range(num_papers, num_papers + base):
            paper_set.add(j)
        graph[num_authors] = paper_set
        num_authors += 1
        num_papers +=  base

    for d in range(1, depth+1): 
        for i in range(nary ** (depth - d)): ### authors at height = d
            paper_set = set()
            for j in range(i * (nary**d) , (i+1) * (nary**d)):
                paper_set.update(graph[j])
            graph[num_authors] = paper_set
            num_authors += 1
        

    return graph, num_authors, num_papers


def tree():
    data_dir = 'data/' + 'tree/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    depth = 3
    base = 2
    nary = 2

    for depth, base, nary  in [(10, 2, 2), (6, 3, 3)]:
        tree,m,n = tree_graph(depth, base, nary)
        print(m, n)
        # continue
        conference_name = f'{nary}nary-{depth}x{base}'
        for review_noise in [1, 2, 3, 4]:
            for self_noise in [0.1, 0.5, 1, 2]:
                for instance in range(5):
                    papers, authors = generate_from_graph(tree, m, n, self_noise=self_noise, review_noise=review_noise)
                    conference = Conference(papers, authors)
                    save_conference(conference, data_dir, '{}-{}-{}-{}'.format(conference_name, review_noise, self_noise, instance))

        res_dir = f'./res/{conference_name}/' 
        if not os.path.exists(res_dir): os.mkdir(res_dir)		

        for level in range(0, depth+1):
            num_part = nary**level
            part_size = n // num_part

            partition = [ set([j for j in range(i*part_size, (i+1)*part_size )]) for i in range(num_part)]
            partition.append(set())
            author_parts = validate(partition, tree, n)

            np.save(f'{res_dir}{level+1}-partition.npy', partition)
            np.save(f'{res_dir}{level+1}-author_parts.npy', author_parts)

    

if __name__ == "__main__":
    gen_from_iclr('iclr21')
    gen_from_iclr('iclr22')
    gen_from_iclr('iclr23')

    tree()

