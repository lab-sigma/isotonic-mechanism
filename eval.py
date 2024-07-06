import numpy as np
import csv, os
from scipy import stats
from sklearn import metrics

from isotonic import *
from partition import *
from instance import *

	
def loss_func(x, y):
	return np.square(x-y).sum()

def precision(ranking1, ranking2, true_scores, k):
	precision_k = len( set(ranking2[:k]) & set(ranking1[:k]) ) / k
	score_k = np.sum( true_scores[ranking2[:k]] )
	return score_k, precision_k

def baseline(conference):
	review_scores = np.asarray( [conference.papers[i]['rev_score'] for i in range(conference.n) ] )
	return review_scores

def benchmark(conference):
	review_scores = np.asarray( [conference.papers[i]['rev_score'] for i in range(conference.n) ] )
	true_scores = np.asarray( [conference.papers[i]['true_score'] for i in range(conference.n) ] )
	rankings = np.argsort(-true_scores)
	calibrated_scores = isotonic(review_scores, rankings)

	return calibrated_scores

def normalised_kendall_tau_distance(values1, values2, coverage=None):
	"""Compute the Kendall tau distance."""
	n = len(values1)
	assert len(values2) == n, "Both lists have to be of equal length"
	i, j = np.meshgrid(np.arange(n), np.arange(n))
	a = np.argsort(values1)
	b = np.argsort(values2)
	if coverage is None:
		ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
	else:
		ndisordered = np.sum( coverage[i,j] * np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])) )

	return ndisordered / (n * (n - 1))

### how many pairs are flipped in true ranking and review ranking
def check_flipped(ranking1, ranking2):
	flipped_pairs = np.zeros((len(ranking1), len(ranking1)))
	for i in range(len(ranking1)):
		for j in range(i+1, len(ranking1)):
			if (ranking1[i] < ranking1[j] and  ranking2[i] > ranking2[j]) or (ranking1[i] > ranking1[j] and ranking2[i] < ranking2[j]):
				flipped_pairs[i][j] = 1
				flipped_pairs[j][i] = 1
	return flipped_pairs

### how many flipped pairs are elicited by the partition
def count_flipped(flipped_pairs, partition): 
	cnt = 0
	for part in partition:
		pp = list(part)
		for i in range(len(part)):
			for j in range(i+1, len(part)):
				if flipped_pairs[pp[i]][pp[j]] == 1:
					cnt += 1
	return cnt

def calibrate(conference, partition, author_parts, mode=None):
	# run isotonic regression for each partition
	calibrated_scores = np.zeros(conference.n)

	for part, author_part in zip(partition, author_parts):
		if len(author_part) == 0:	
			for i in part: calibrated_scores[i] = conference.papers[i]['rev_score']
			continue

		paper_part = list(part) # fix the order of papers
		review_scores = np.asarray( [conference.papers[i]['rev_score'] for i in paper_part ] )

		rankings = []
		for author in author_part:
			scores = np.asarray( [conference.authors[author][i] for i in paper_part] )
			rankings.append(np.argsort(-scores))

		if mode == None:
			scores = isotonic(review_scores, rankings[0])
		elif mode == 'multi':
			scores = multiisotonic(review_scores, rankings)
		elif mode == 'soft':
			# calibrate scores
			scores = softisotonic(review_scores, rankings)
			
		for i, score in zip(paper_part, scores): calibrated_scores[i] = score

	return calibrated_scores

def test(conference, mode='greedy', save_dir=None):
	graph = conference.graph
	m = conference.m
	n = conference.n

	if save_dir is not None and os.path.exists(save_dir + mode + '-partition.npy'):
		partition = np.load(save_dir + mode +  '-partition.npy', allow_pickle=True)
		author_parts = np.load(save_dir + mode + '-author_parts.npy', allow_pickle=True)
	else:
		if mode == 'greedy-1':
			partition, author_parts = greedy(graph, m, n)
			author_parts = validate(partition, graph, n)
		elif mode == 'arbitrary':
			partition, author_parts = arbitrary(graph, m, n, randomize=True)

		if save_dir is not None:
			print(f'saving partition by {mode}')
			print(partition)
			np.save(save_dir + mode + '-partition.npy', partition)
			np.save(save_dir + mode + '-author_parts.npy', author_parts)
			
	
	calibrated_scores = calibrate(conference, partition, author_parts, mode='multi')
	
	return calibrated_scores, partition


def stats(conference, partition, true_scores, true_ranking, calibrated_scores, base, mode, writer):
	loss = loss_func(calibrated_scores, true_scores) / len(true_scores) 
	total_pairs = sum( len(part) * (len(part) - 1)  for part in partition ) // 2
	num_flipped = count_flipped(flipped_pairs, partition)

	calibrated_ranking = np.argsort(-calibrated_scores)
	k1 = int(conference.n * 0.015)
	k5 = int(conference.n * 0.05)
	k30 = int(conference.n * 0.3)
	score_1, precision_1 = precision(true_ranking, calibrated_ranking, true_scores, k=k1)
	score_5, precision_5  = precision(true_ranking, calibrated_ranking, true_scores, k=k5)
	score_30, precision_30,  = precision(true_ranking, calibrated_ranking, true_scores, k=k30)

# 'review_noise', 'mode', 'loss', '#elicited pairs', '#elicited-flipped pairs', 'score-1.5', 'precision-1.5', 'score-5', 'precision-5', 'score-30', 'precision-30'
	writer.writerow( [*base, mode, loss, total_pairs, num_flipped, score_1, precision_1, score_5, precision_5, score_30, precision_30])


	return calibrated_scores



if __name__ == "__main__":

	for data_name in ['iclr21', 'iclr22', 'iclr23']:
		res_dir = f'./res/{data_name}-2/' 
		data_dir = f'./data/{data_name}/' 
		if not os.path.exists(res_dir):
			os.mkdir(res_dir)		

		csv_file = open(res_dir + 'results.csv', 'a')
		writer = csv.writer(csv_file)

		# precision: the number of accepted papers that are indeed good
		# score: total score of accepted papers
		# oral, spotlight, poster: 1.5, 5, 30
		writer.writerow(['review_noise', 'self_noise', 'instance', 'mode', 'loss', '#elicited pairs', '#elicited-flipped pairs', 'score-1.5', 'precision-1.5', 'score-5', 'precision-5', 'score-30', 'precision-30'])

		for review_noise in [1, 2, 3, 4]:
			for self_noise in [0, 0.1, 0.5, 1, 2]:
				for instance in range(5):
					base = [review_noise, self_noise, instance]
					conference = load_conference(data_dir, f'{data_name}-{review_noise}-{self_noise}-{instance}')

					review_scores = np.asarray( [conference.papers[i]['rev_score'] for i in range(conference.n) ] )
					true_scores = np.asarray( [conference.papers[i]['true_score'] for i in range(conference.n) ] )
					true_ranking = np.argsort(-true_scores)
					review_ranking = np.argsort(-review_scores)

					# print( true_scores[ np.argmax(true_ranking) ], true_scores[ np.argmin(true_ranking) ] )
					# print( true_scores[ true_ranking[0] ], true_scores[ true_ranking[-1] ] )
					# print( np.max(true_scores), np.min(true_scores) )

					flipped_pairs = check_flipped(true_ranking, review_ranking)

					# scores = baseline(conference)
					# writer.writerow([review_noise, 'baseline', loss, 0, 0, ])	
					### baseline
					stats(conference, [], true_scores, true_ranking, review_scores, base, 'baseline', writer)
					# ### benchmark
					# calibrated_scores = benchmark(conference)
					# stats(conference, [[i for i in range(conference.n)]], true_scores, true_ranking, calibrated_scores, base, 'benchmark', writer)
					### greedy partition
					calibrated_scores, partition = test(conference, mode='greedy-1', save_dir=res_dir )
					stats(conference, partition[:-1], true_scores, true_ranking, calibrated_scores, base, 'greedy-1', writer)
					### random partition
					calibrated_scores, partition = test(conference, mode='arbitrary', save_dir=res_dir )
					stats(conference, partition[:-1], true_scores, true_ranking, calibrated_scores, base, 'arbitrary', writer)
				break
			break
		csv_file.close()

	for depth, base, nary  in [(10, 2, 2), (6, 3, 3)]:
		data_name = f'{nary}nary-{depth}x{base}'
		res_dir = f'./res/{data_name}/' 
		data_dir = f'./data/tree/' 
		if not os.path.exists(res_dir):
			os.mkdir(res_dir)		

		csv_file = open(res_dir + 'results.csv', 'w')
		writer = csv.writer(csv_file)

		# precision: the number of accepted papers that are indeed good
		# score: total score of accepted papers
		# oral, spotlight, poster: 1.5, 5, 30
		writer.writerow(['review_noise', 'self_noise', 'instance', 'mode', 'loss', '#elicited pairs', '#elicited-flipped pairs', 'score-1.5', 'precision-1.5', 'score-5', 'precision-5', 'score-30', 'precision-30'])

		for review_noise in [1, 2, 3, 4]:
			for self_noise in [0.1, 0.5, 1, 2]:
				for instance in range(5):
					base = [review_noise, self_noise, instance]
					print( review_noise, self_noise, instance )
					conference = load_conference(data_dir, f'{data_name}-{review_noise}-{self_noise}-{instance}')

					review_scores = np.asarray( [conference.papers[i]['rev_score'] for i in range(conference.n) ] )
					true_scores = np.asarray( [conference.papers[i]['true_score'] for i in range(conference.n) ] )
					true_ranking = np.argsort(-true_scores)
					review_ranking = np.argsort(-review_scores)

					flipped_pairs = check_flipped(true_ranking, review_ranking)

					# scores = baseline(conference)
					# writer.writerow([review_noise, 'baseline', loss, 0, 0, ])	
					### baseline
					stats(conference, [], true_scores, true_ranking, review_scores, base, 'baseline', writer)
					### benchmark
					calibrated_scores = benchmark(conference)
					stats(conference, [[i for i in range(conference.n)]], true_scores, true_ranking, calibrated_scores, base, 'benchmark', writer)

					### greedy partition with different L-strongness 
					for i in range(1, depth+2):
						calibrated_scores, partition = test(conference, mode=str(i), save_dir=res_dir )
						stats(conference, partition[:-1], true_scores, true_ranking, calibrated_scores, base, f'greedy-{i}', writer)
					
		csv_file.close()

