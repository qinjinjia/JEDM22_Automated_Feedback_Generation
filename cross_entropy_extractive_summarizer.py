'''
Created on Jun 19, 2021

This script uses the Cross-Entropy Method to summarize the text extracted from
the Expertiza Wiki pages.
'''


# Imports
################################################################################
import argparse
import json
import numpy as np
import os
import re
import string
from nltk import MWETokenizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer



# Settings
################################################################################
#------------------------------------------------------------------------------#
# stop words, punctuation, and extraneous words to be removed for CE method
stop_words = set(stopwords.words('english'))
punctuation = string.punctuation
words_to_ignore = set([
	"''",'""',"``"
])
#------------------------------------------------------------------------------#
# Multi-Word-Tokenizer for retaining special tokens when word splitting
mwtokenizer = MWETokenizer(separator='')
multi_word_expressions = [
    ("<","image",">"),
    ("<","code",">"),
    ("<","link",">"),
]
for mwe in multi_word_expressions:
	mwtokenizer.add_mwe(mwe);
#------------------------------------------------------------------------------#
# regex for finding sentences split by numbered bullets (Ex. 1.4.2.)
sentence_number_reg = re.compile(".*?\d+\.$$")
# regex for removing numbered bullets in sentence
word_number_reg = re.compile("[0-9]\.?[0-9\.]*")
#------------------------------------------------------------------------------#
# pre-trained Bert tokenizer for determining length of summaries (since they need to fit into BERT model)
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
# add special tokens
special_tokens = [
    "<link>",
    "<image>",
    "<table>",
    "<code>",
]
special_tokens_dict = {'additional_special_tokens': special_tokens}
bart_tokenizer.add_special_tokens(special_tokens_dict)



# Helper Functions
################################################################################
def textLength(full_text):
	"""Gets number of tokens used to encode text for BART model."""
	return len(bart_tokenizer(full_text)["input_ids"])
#------------------------------------------------------------------------------#
def passesLengthLimit(full_text):
	"""Determines if full text can be passed into BART model after tokenization."""
	return len(bart_tokenizer(full_text)["input_ids"]) <= 1024
#------------------------------------------------------------------------------#
def json2Text(data, use_headers=False):
	"""
	Takes the json object representing the scraped data from an Expertiza Wiki
	page and parses out the text. In doing so, we have the option of either
	keeping or removing the section headers from the text. Returns the parsed
	test as a single continuous string.

	Args:
			data (json-like dictionary): json object containing scraped Wiki data
			use_headers (bool, optional): Determines if section headers should be
				part of returned text. Defaults to False.

	Returns:
			str: the parsed text from our json data
	"""
	# grab text from this object and store in list, using headers if desired
	if use_headers:
		text = data["title"].strip()
		# place colon at end to indicate it is a header
		if not text[-1] == ':':
			text += ':'
		# append actual non-title text
		if data["text"]:
			text += ' ' + data["text"].strip()
	else:
		text = (data["text"] or "").strip()

	# ensure text ends in a period (to indicate new sentence)
	if len(text) > 0 and text[-1] not in [':','.']:
		text += '.'

	# recursively grab text from this objects children and append to text
	if data["children"]:
		for child_data in data["children"]:
			child_text = json2Text(child_data, use_headers)
			if len(child_text) > 0:
				text += ' ' + child_text

	# return resulting text string
	return text.strip()

#------------------------------------------------------------------------------#
def tokenizeSentences(text):
	"""
	Separates given text into list of sentence tokens. Primarily, this is done
	using NLTK's built in sentence tokenizer. However, a slight modification is
	made to recombine sentences that are split due to numbered bullets within
	(Ex: "1.4.3" which NLTK will identify as the end of a sentence due to the
	period, even though it is not).

	Args:
			text (str): full text that we wish to summarize

	Returns:
			str[]: list of sentence tokens
	"""
	# use nltk sentence tokenizer to break up sentences on punctuation
	sentences = sent_tokenize(text)
	# recombine sentences split because of numbered bullets (Ex: 1.3.)
	sentence_tokens = []
	n = len(sentences)
	if n == 0:
		return []
	sentence = sentences[0]
	for i in range(n):
		if re.match(sentence_number_reg, sentence) and i<n-1:
			sentence = sentence + " " + sentences[i+1]
		else:
			sentence_tokens.append(sentence)
			if i<n-1:
				sentence = sentences[i+1]
	# return sentence tokens
	return sentence_tokens

#------------------------------------------------------------------------------#
def tokenizeWords(sentence_tokens):
	"""
	Splits sentence tokens into more granular word tokens. In doing so, this
	function also parses the text by removing stop words, punctuation, and other
	extraneous tokens we do not wish to include in our CE method. Also of note,
	in this function we specify our special tokens (Ex: "<image>") which we want
	our tokenizer not to split up using the built in NLTK MWETokenizer.

	Args:
			sentence_tokens (str[]): list of sentence tokens

	Returns:
			str[][]: list of sentences, each sentence being a list of parsed word tokens
	"""
	
	# iterate over sentences, tokenizing each into word tokens
	word_tokens = []
	for sentence in sentence_tokens:
		# first remove numbered bullets from sentences and make lowercase
		parsed_sentence = re.sub(word_number_reg, "", sentence).lower()
		# next tokenize sentences into words, being sure to retain our special tokens
		words = mwtokenizer.tokenize(word_tokenize(parsed_sentence))
		# lastly, remove all stop words, punctuation, and other extraneous tokens
		filtered_words = []
		for w in words:
			if not (w in stop_words or w in punctuation or w in words_to_ignore):
				filtered_words.append(w)
		# push our newly tokenized sentence
		word_tokens.append(filtered_words)
	# return all tokenized sentences
	return word_tokens

#------------------------------------------------------------------------------#
def word_freq(S):
	"""
	Creates a mapping of the frequencies of all word tokens in the given summary
	S. This is used in our the performance funciton of our CE method, specifically
	for calculating the value of Quality Feature 1: Unigram Diversity of Summary.

	Args:
			S (str[][]): summary of text consisting of list of sentences, with each sentence being a list of word tokens

	Returns:
			dict: python dictionary mapping word_token: str -> frequency: float
	"""
	# initialize variables to count words
	total_words = 0
	word_freqs = {}

	# loop over all words keeping track of their frequencies
	for sentence in S:
		for word in sentence:
			total_words += 1
			try:
				word_freqs[word] += 1
			except:
				word_freqs[word] = 1
	
	# divide frequencies by total # of words
	for word in word_freqs:
		word_freqs[word] = word_freqs[word] / total_words

	# return resulting word frequencies
	return word_freqs

#------------------------------------------------------------------------------#
def performanceFunction(S, full_text, include_length=True):
	"""Calculates the performance function value R(S) for a given summary S. These
	values are used directly in the CE method."""

	# if too many tokens, return -Inf, else actually compute R
	num_tokens = textLength(full_text)
	if num_tokens > 1024:
		return -np.inf

	# initialize variable to store performance
	R = 0

	# Feature Quality 1: Unigram Diversity of Summary
	# -----------------------------------------------
	# get word frequencies
	p_S = word_freq(S)

	# iterate over words adding to performance
	for sentence in S:
		for word in sentence:
			R -= p_S[word] * np.log(p_S[word])

	# Feature Quality 2: Total Length of Summary
	# ------------------------------------------
	# adjust for length if desired
	if include_length:
		R += 5 * num_tokens / 1024

	# return performance meausure
	return R

def filter2Largest(array, N):
	"""Function for filtering given array to just its N largest values."""
	if N >= len(array):
		return array.astype("bool")
	quantile = 1 - (N / len(array))
	quantile_value = np.quantile(array, quantile)
	return array >= quantile_value

def finalSample(p, sentence_tokens):
	"""Grabs a sample summary of our text using the final p values generated
	using the CE method. In doing so, we make sure that the summary meets our
	length criteria (less than 1024 tokens). Also returns some stats on the
	positions and actual probabilities associated with the chosen summary
	sentences."""
	# take final sample from p, attempting until sample is under length limit
	while True:
		x = np.random.binomial(1,p).astype("bool")
		summarized_text = " ".join(np.array(sentence_tokens)[x])
		if passesLengthLimit(summarized_text):
			break

	# grab other stats
	sentence_positions = (np.arange(len(x))[x] / len(x)).tolist()
	sentence_probabilities = p[x].tolist()

	# return everything
	return sentence_positions , sentence_probabilities , summarized_text




# Main Export/Functionality
################################################################################
def crossEntropySummarizer(text, sentence_limit=30, N=1000, rho=.05, alpha=.7, no_change_limit = 3):
	"""
	Summarizes a given string of text to meet our length criteria (less than 1024
	BART tokens) using the CE method. Note that many of the parameters for the
	method are inputs to this function and not the batch summarizer below. Simply
	change these defaults to your desired values for tesing and experimenting
	(by default I made them the same values as used in the paper which seemed
	to work reasonably well, though increasing N and decreasing rho will produce
	slightly better results if you are okay with the runtime increasing).

	Args:
			text (str): text to be summarized
			sentence_limit (int, optional): Specifies roughly the number of sentences 
					we need to filter to. Only used for initializing p values for samples.
					Defaults to 30.
			N (int, optional): Summary sample size per CE iteration. Defaults to 1000.
			rho (float, optional): Upper quantile used to designate the Elite Sample. 
					Defaults to .05.
			alpha (float, optional): Value for smoothing CE updates. Value should
					be in range [0,1] where large alpha yields less smoothing of updates.
					Defaults to .7.
			no_change_limit (int, optional): Essentially the termination parameter.
					This is the number of successive time the gamma value must remain the
					same before terminating the CE method (since this indicates no change
					to our results is occuring). Defaults to 3.

	Returns:
			tuple: (sentence_positions , sentence_probabilities , summarized_text)
					tuple including summarized text, the positions of the chosen setences,
					and their probabilities of selection
	"""

	# tokenize text into sentences
	sentence_tokens = tokenizeSentences(text)

	# if text already passes text limit then just return text as is
	if (passesLengthLimit(text)):
		was_summarized = False
		summarized_text = text
		sentence_positions = (np.arange(len(sentence_tokens)) / len(sentence_tokens)).tolist()
		sentence_probabilities = [1 for i in range(len(sentence_tokens))]
	# else run the CE method
	else:
		was_summarized = True

		# tokenize sentences by word (removing stop words and making lowercase)
		word_tokens = np.array(tokenizeWords(sentence_tokens), dtype=object)

		# grab number of sentences in text
		n = len(sentence_tokens)

		# initialize variables for method
		if n>sentence_limit*2:
			p = np.full(n, sentence_limit / n)
		else:
			p = np.full(n, .5)
		gamma_temp = 0
		gamma = 0
		no_change_count = 0
		
		# iteratively sample and update our summary until it converges
		# TODO: remove counter once working properly (currently prevents infinite loops)
		iterations = 0
		while no_change_count < no_change_limit:
			iterations += 1

			# grab samples, ensuring we get N samples all under the sentence limit
			X = np.random.binomial(1,p,size=(N,n)).astype("bool")

			# grab performance measures for each sample
			R = np.zeros(N)
			for i in range(N):
				S = word_tokens[X[i]]
				full_text = " ".join(np.array(sentence_tokens)[X[i]])
				R[i] = performanceFunction(S, full_text)

			# calculate gamma (the 1-rho percentile value from R)
			gamma = np.percentile(R, 100*(1-rho))
			print("Iteration {}: gamma = {}".format(iterations,gamma))

			# see if gamma has change since last iteration
			if (gamma == gamma_temp):
				no_change_count += 1
			else:
				no_change_count = 0
			gamma_temp = gamma

			# calculate p-hat
			I_R = R >= gamma
			D = I_R.sum()

			p_hat = np.dot(I_R, X.astype("int")) / D

			# update p for next iteration
			p = alpha*p_hat + (1-alpha)*p
			
		# sample final p vector to determine which sentences to include in summary
		sentence_positions , sentence_probabilities , summarized_text = finalSample(p, sentence_tokens)

	

	# return results
	summary_stats = {
		"was_summarized": was_summarized,
		"sentence_positions": sentence_positions,
		"sentence_probabilities": sentence_probabilities,
	}
	return summarized_text , summary_stats
	


def batchSummarizer(source_dir, output_dir, limit=None):
	"""
	Wrapper function used to run the Cross Entropy Summarizer function above on
	all files from a designated source directory. All files in the source
	directory must be JSON files with the exact format expected from the
	output of the Wiki Web Scraper. Each CE method summary is output to a JSON
	file in the designated output directory.

	Additionally, a limit can be specified so that the batch summarizer only runs
	on a set number of total files. This was useful for testing on only a small
	number of examples at a time.

	Args:
			source_dir (str): path to directory containing source JSON files (Wiki Web Scraper Output)
			output_dir (str): path to directory to save CE method output
			limit (int, optional): Limits the number of files to be summarized. Defaults to None.
	"""

	# first grab all the json files in the source directory
	json_files = [file_name for file_name in os.listdir(source_dir) if file_name.endswith('.json')]

	# iterate over all files, counting word frequencies in each
	file_count = 0
	for file_name in json_files:

		# open file and grab the data
		# file_name = "E1405.json"
		with open(os.path.join(source_dir,file_name),'r') as json_file:
			data = json.load(json_file)

		# parse string of text from data and make it all lowercase
		text = json2Text(data["data"], use_headers=False)

		# print message to indicate we are summarizing another file
		message = "Summarizing File: {}".format(file_name)
		print()
		print(message)
		print(len(message)*"=")

		# use cross entropy method to determine which sentences to include in summary
		summarized_text , summary_stats = crossEntropySummarizer(text)

		# output summarized text to output json file
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		with open(os.path.join(output_dir, file_name.replace(".json","_summary.json")), 'w') as output_file:
			json_data = {
				"project_id": data["project_id"],
				"unique_id": data["unique_id"],
				"title": data["data"]["title"],
				"original_text": text,
				"summarized_text": summarized_text,
				"summary_stats": summary_stats,
			}
			json.dump(json_data, output_file, indent=4)

		# update file count
		file_count += 1
		if limit and file_count >= limit:
			break



# Argparser & Main
################################################################################
def parse_arguments():
	"""
	Parser command line arguments and displays help text.

	Returns:
			args: parsed arguments passed into script through command line
	"""

	parser = argparse.ArgumentParser(description="Summarizes each the text from"
		" each of our extracted json files using the Cross-Entropy method."
	)
	parser.add_argument(
		"source_dir",
		help="path to directory containing the extracted json data",
		type=str
	)
	parser.add_argument(
		"output_dir",
		help="directory to write summarized text files to",
		type=str
	)
	parser.add_argument(
		"-l",
		"--limit",
		help='limit on number of json files to summarize (usually for testing)',
		type=int
	)
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_arguments()
	batchSummarizer(args.source_dir, args.output_dir, args.limit)


################################################################################
# FIN # FIN # FIN # FIN # FIN # FIN # FIN # FIN # FIN # FIN # FIN # FIN # FIN #
################################################################################