from gensim import models

from random import sample
from gensim.matutils import kullback_leibler, hellinger, jaccard_set, WMD
from gensim.models.ldamulticore import LdaMulticore
import numpy as np



def topic2topic_difference(m1, m2, distance="kulback_leibler", num_words=100, topw=10, with_annotation=True, model=None):
    """
    Calculate difference topic2topic between two `LdaMulticore` models

    `m1` and `m2` are trained instances of `LdaMulticore`
    `distance` is function that will be applied to calculate difference between any topic pair.
    Available values: `kulback_leibler`, `hellinger` and `jaccard`
    `num_words` is quantity of most relevant words that used if distance == `jaccard` (also used for annotation)
    `topw` is max quantity for positive/negative words

    Returns a matrix Z with shape (m1.num_topics, m2.num_topics), where Z[i][j] - difference between topic_i and topic_j

    If `with_annotation` == True, additionally return array with shape (m1.num_topics, m2.num_topics, 2, None)
    where
        Z[i][j] = [[`pos_1`, `pos_2`, ...], [`neg_1`, `neg_2`, ...]] and
        `pos_k` is word from intersection of `topic_i` and `topic_j` and
        `neg_l` is word from symmetric difference of `topic_i` and `topic_j`


    Example:

    >>> m1, m2 = LdaMulticore.load(path_1), LdaMulticore.load(path_2)
    >>> mdiff, annotation = topic2topic_difference(m1, m2)
    >>> print(mdiff) # get matrix with difference for each topic pair from `m1` and `m2`
    >>> print(annotation) # get array with positive/negative words for each topic pair from `m1` and `m2`
    """

    distances = {"kulback_leibler": kullback_leibler,
                 "hellinger": hellinger,
                 "jaccard": jaccard_set,
                 "WMD" : WMD}

    assert distance in distances, "Incorrect distance, valid only {}".format(", ".join("`{}`".format(x)
                                                                                       for x in distances.keys()))
    assert isinstance(m1, LdaMulticore), "The parameter `m1` must be of type `{}`".format(LdaMulticore.__name__)
    assert isinstance(m2, LdaMulticore), "The parameter `m2` must be of type `{}`".format(LdaMulticore.__name__)

    distance_func = distances[distance]
    d1, d2 = m1.state.get_lambda(), m2.state.get_lambda()
    t1_size, t2_size = d1.shape[0], d2.shape[0]

    fst_topics, snd_topics = None, None

    if distance == "jaccard" or distance == 'WMD':
        d1 = fst_topics = [{w for (w, _) in m1.show_topic(topic, topn=num_words)} for topic in range(t1_size)]
        d2 = snd_topics = [{w for (w, _) in m2.show_topic(topic, topn=num_words)} for topic in range(t2_size)]

    z = np.zeros((t1_size, t2_size))

    for topic1 in range(t1_size):
        for topic2 in range(t2_size):
            if distance == 'WMD':
                z[topic1][topic2] = distance_func(d1[topic1], d2[topic2], model)
            else:
                z[topic1][topic2] = distance_func(d1[topic1], d2[topic2])

    z /= np.max(z)

    if not with_annotation:
        return z

    annotation = [[None for _ in range(t1_size)] for _ in range(t2_size)]

    if fst_topics is None or snd_topics is None:
        fst_topics = [{w for (w, _) in m1.show_topic(topic, topn=num_words)} for topic in range(t1_size)]
        snd_topics = [{w for (w, _) in m2.show_topic(topic, topn=num_words)} for topic in range(t2_size)]

    for topic1 in range(t1_size):
        for topic2 in range(t2_size):
            if topic2 < topic1:
                continue

            pos_tokens = fst_topics[topic1] & snd_topics[topic2]
            neg_tokens = fst_topics[topic1].symmetric_difference(snd_topics[topic2])

            pos_tokens = sample(pos_tokens, min(len(pos_tokens), topw))
            neg_tokens = sample(neg_tokens, min(len(neg_tokens), topw))

            annotation[topic1][topic2] = annotation[topic2][topic1] = [pos_tokens, neg_tokens]

    return z, annotation



def score_difference(lda1, lda2, distance="kulback_leibler", num_words=100,model=None):
    topic_similarity_array = topic2topic_difference(lda1, lda2,
                                                    distance=distance,
                                                    num_words=num_words,
                                                    with_annotation=False,
                                                    model=model)
    sorted_array = sort_topic_similarity(topic_similarity_array)
    topic_pairs_similarity = pair_topics(topic_similarity_array, sorted_array,
                                         sort_flag=True)

    sorted_output = list(topic_pairs_similarity)
    bijections = match_topics_bijection(sorted_output, sorted_array,
                                        topic_similarity_array)
    score = bijection_score(bijections)
    return score, bijections


def sort_topic_similarity(topic_similarity_array):
    """
    Given an N x N array of topic similarity scores (returned by 
    topic2topic_difference), this function sorts these scores by row and
    returns the sorted indices

    :param topic_similarity_array: a numpy array returned by 
        topic2topic_difference
    :return: 
        returns an N x N array of sorted indices of topic_similarity_array by
        row
    """
    return np.argsort(topic_similarity_array)


def pair_topics(topic_similarity_array, sorted_array, sort_flag=False):
    """
    Given an N x N array of topic similarity scores (which is returned
    by topic2topic_difference), and an N x N array of these scores' indices 
    sorted by row (returned by sort_topic_similarity), this function matches 
    each topic with the most similar topic to it. It creates a tuple of each
    pair and their respective similarity. This function allows multiple
    topics to be matched together: it does not enforce a 1 to 1 relationship

    :param topic_similarity_array: a numpy array returned by 
        topic2topic_difference
    :param sorted_array: a numpy array returned by sort_topic_similarity
    :param sort_flag: boolean of whether to sort matched pairs by similarity
    :return: 
        topic_similarity_scores: a list of booleans representing each topic
            pairing and their score 
    """
    topic_similarity_scores = []

    for x in range(0, len(topic_similarity_array)):
        most_similar_topic = sorted_array[x][0]
        pair = (x, most_similar_topic)

        score = topic_similarity_array[x][most_similar_topic]
        topic_similarity_scores.append((score, pair))

    if sort_flag:
        topic_similarity_scores.sort()

    return topic_similarity_scores


def match_topics_bijection(pair_list, sort_array, topic_similarity_array):
    """
    Enforces a strict 1-to-1 bijection of each models' topics. It begins by
    matching the most similar topics. If a topic has already been matched, the
    next most similar topic pairing is used, and the list of pairings in 
    pair_list is re-sorted.

    :param pair_list: sorted list of topic pairs and their similarity
    :param sort_array: the sorted indices of a topic_similarity_array. It is
        needed to find the next most similar pair in case of collision.
    :param topic_similarity_array: a numpy array returned by 
        topic2topic_difference
    :return: 
        output: a sorted list of matched topics
    """
    match_topics = []
    included_topics = []

    while len(pair_list) > 0:
        most_similar_pair = pair_list.pop(0)
        topic1 = most_similar_pair[1][0]
        topic2 = most_similar_pair[1][1]

        if topic2 in included_topics:
            n = 0
            while topic2 in included_topics:
                n += 1
                topic2 = sort_array[topic1][n]

            pair = (topic1, topic2)
            score = topic_similarity_array[topic1][topic2]

            next_most_similar_pair = (score, pair)
            pair_list.append(next_most_similar_pair)
            pair_list.sort()
        else:
            included_topics.append(topic2)
            match_topics.append(most_similar_pair)
    return match_topics

def bijection_score(bijections):
    total_score = 0

    for score_tuple in bijections:
        total_score += score_tuple[0]

    score_average = total_score / len(bijections)
    return score_average

