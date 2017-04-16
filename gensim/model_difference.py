from random import sample
from gensim.matutils import kullback_leibler, hellinger, jaccard_set
from gensim.models.ldamulticore import LdaMulticore
import numpy as np


def topic2topic_difference(m1, m2, distance="kulback_leibler", num_words=100, topw=10, with_annotation=True):
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
                 "jaccard": jaccard_set}

    assert distance in distances, "Incorrect distance, valid only {}".format(", ".join("`{}`".format(x)
                                                                                       for x in distances.keys()))
    assert isinstance(m1, LdaMulticore), "The parameter `m1` must be of type `{}`".format(LdaMulticore.__name__)
    assert isinstance(m2, LdaMulticore), "The parameter `m2` must be of type `{}`".format(LdaMulticore.__name__)

    distance_func = distances[distance]
    d1, d2 = m1.state.get_lambda(), m2.state.get_lambda()
    t1_size, t2_size = d1.shape[0], d2.shape[0]

    fst_topics, snd_topics = None, None

    if distance == "jaccard":
        d1 = fst_topics = [{w for (w, _) in m1.show_topic(topic, topn=num_words)} for topic in range(t1_size)]
        d2 = snd_topics = [{w for (w, _) in m2.show_topic(topic, topn=num_words)} for topic in range(t2_size)]

    z = np.zeros((t1_size, t2_size))

    for topic1 in range(t1_size):
        for topic2 in range(t2_size):
            if topic2 < topic1:
                continue

            z[topic1][topic2] = z[topic2][topic1] = distance_func(d1[topic1], d2[topic2])

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
