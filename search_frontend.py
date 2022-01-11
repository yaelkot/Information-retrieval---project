import re
from collections import Counter
import requests
from google.cloud import storage
from flask import Flask, request, jsonify
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import math
from itertools import chain
import time
import csv
import re
from contextlib import closing
import csv



from inverted_index_gcp import MultiFileReader
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)



app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False




def posting_lists_iter_read(index,path):
    """ A generator that reads one posting list from disk and yields
        a (word:str, [(doc_id:int, tf:int), ...]) tuple.
    """
    with closing(MultiFileReader()) as reader:
        for w, locs in index.posting_locs.items():
            b = reader.read(locs, index.df[w] * TUPLE_SIZE, path)
            posting_list = []
            for i in range(index.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            yield w, posting_list

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer
from contextlib import closing


"""
    :param inverted - inverted index
    :param w - word
    :param path - given path to the specify inverted index
    function that read and return posting list from inverted index with thw given word
"""
def read_posting_list(inverted, w,path):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE,path)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list

"""
    function that calculate cosine similarity between query and words in inverted index
"""
def cos_sim(query, index,path):
    dict_q_sim = {}
    sim_q_score = 0
    q_term_weight = {}
    for word in query:
        q_term_weight[word] = query.count(word)
    for doc_id in index.DL.keys():
        dict_q_sim[doc_id] = sim_q_score
    for term in query:
        if term in index.posting_locs:
            posting_list = read_posting_list(index, term,path)

        for (doc_id, weight_in_doc) in posting_list:
            if doc_id != 0:
              dict_q_sim[doc_id] += q_term_weight[term] * weight_in_doc
    for doc_id in index.DL.keys():
        if index.DL[doc_id] != 0 and len(query)!=0:
            dict_q_sim[doc_id] = dict_q_sim[doc_id] * (1 / len(query) * (1 / index.DL[doc_id]))
    return dict_q_sim


"""
    function that calculate similarity between all the query and all the titles
"""
def cos_sim_title(query):
    dict_q_sim = {}

    for word in query:
        if word in data_title.words_titles:
            for doc_id in data_title.words_titles[word]:
                if doc_id not in dict_q_sim:
                    dict_q_sim[doc_id] = 1/len(data_title.titles[doc_id])
                else:
                    dict_q_sim[doc_id] += 1/len(data_title.titles[doc_id])

    return dict_q_sim


####################################################
########## Loading all inverted index ##############
####################################################

with open('postings_gcp_text/index_body.pkl', 'rb') as f:
    data_body = pickle.load(f)

with open('postings_gcp_title/index_title.pkl', 'rb') as f:
    data_title = pickle.load(f)


with open('postings_gcp_anchor/index_anchor.pkl', 'rb') as f:
    data_anchor = pickle.load(f)


with open('postings_gcp_anchor/pv_clean.pkl', 'rb') as f:
    wid2pv = pickle.load(f)


with open('pr/part-00000-3043c3f3-47de-4ed8-933a-f323183201d9-c000.csv', mode='r') as infile:
    reader = csv.reader(infile)
    pr_dict = {int(rows[0]):float(rows[1]) for rows in reader}



#######################################################
english_stopwords = frozenset(stopwords.words('english'))
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


"""
    function that calculate how many terms from query are in the title
"""
def func(title,query):
    title=title.lower().split()
    num=0
    for i in query:
        if i in title:
            num+=1
    return num

"""
    function that return top N documents form similarity dictionary
"""
def get_top_n(sim_dict, N=3):
    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


"""
    function that merge all the scores with the given weights for each part
"""
def merge_results(title_scores, body_scores,  title_weight=0.5, text_weight=0.5, N=3):

    new_b={}
    for a, b in body_scores.items():
        new_b[a] = b * text_weight
    new_t = {}
    for a, b in title_scores.items():
        new_t[a] = b * title_weight
    return dict(Counter(new_t) + Counter(new_b))

"""
    :param query - given query
    function that gets page view for each page that match to the query and save it in dictionary
"""
def pview(query):
    score={}
    for i in query:
        if i in data_title.words_titles:
            for doc_id in data_title.words_titles[i]:
                if doc_id not in score:
                    if doc_id in pr_dict:
                        score[doc_id]=pr_dict[doc_id]/10
                else:
                    if doc_id in pr_dict:
                        score[doc_id]+=pr_dict[doc_id]/10

"""
    :param query - given query
    function that gets page rank for each page that match to the query and save it in dictionary
"""
def prank(query):
    score={}
    for i in query:
        if i in data_title.words_titles:
            for doc_id in data_title.words_titles[i]:
                if doc_id not in score:
                    if doc_id in wid2pv:
                        score[doc_id]=wid2pv[doc_id]/10000
                else:
                    if doc_id in wid2pv:
                        score[doc_id]+=wid2pv[doc_id]/10000

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION


    spl=[token.group() for token in RE_WORD.finditer(query.lower())]
    spl = [term for term in spl if term not in english_stopwords]
    b=cos_sim(spl,data_body,'postings_gcp_text')
    t=cos_sim_title(spl)
    pv_score=pview(spl)
    pr_score=prank(spl)
    #a=cos_sim(spl,data_anchor,'postings_gcp_anchor')
    wt, wb, wa = 6, 2, 0.5
    score = dict(Counter(merge_results(t, b,  wt, wb, 100))+Counter(pv_score)+Counter(pr_score))
    lst = [(k, score[k]) for k in sorted(score, key=score.get, reverse=True)]
    for i in range(len(lst)):
        if len(res) == 100:
            break
        if (lst[i][0] in data_title.titles):
            res.append((lst[i][0],data_title.titles[lst[i][0]]))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query=[token.group() for token in RE_WORD.finditer(query.lower())]
    query = [term for term in query if term not in english_stopwords]

    helper = cos_sim(query, data_body,'postings_gcp_text')
    lst = get_top_n(helper, 100)
    for i in lst:
        if i[0] in data_title.titles:
            res.append((i[0],data_title.titles[i[0]]))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    query = [token.group() for token in RE_WORD.finditer(query.lower())]
    query = [term for term in query if term not in english_stopwords]
    val=[]
    for i in query:
        if(i in data_title.words_titles):
            val += data_title.words_titles[i]
    count=Counter(val)
    a = count.most_common()
    for i in a:
        if(i[0] in data_title.titles):
            res.append((i[0],data_title.titles[i[0]]))

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")

def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    query = [token.group() for token in RE_WORD.finditer(query.lower())]
    query = [term for term in query if term not in english_stopwords]
    val=[]
    for i in query:
        pls=read_posting_list(data_anchor,i, 'postings_gcp_anchor')
        val+=[t[0] for t in pls]
    count=Counter(val)
    a = count.most_common()
    for j in query:
        for i in a:
            if(i[0] in data_title.titles) and j in data_title.titles[i[0]]:
                res.append((i[0],data_title.titles[i[0]]))
    res=set(res)
    res=list(res)
    res=sorted(res,key=lambda x:func(x[1],query),reverse=True)
    print(type(res))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for i in wiki_ids:
        if i in pr_dict:
            res.append(pr_dict[i])

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for i in wiki_ids:
        if i in wid2pv:
            res.append(wid2pv[i])
    # END SOLUTION
    return jsonify(res)




if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
