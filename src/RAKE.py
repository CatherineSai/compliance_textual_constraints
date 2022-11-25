import pandas as pd
from rake_nltk import Rake
import re
import os

path = os.path.dirname(os.path.abspath(__file__))

def RAKE_Keyword_Extraction(file_type, threshold, stop_word_type):
    """ RAKE keyword extraction method

    :param file_type: input dataset(GDPR or REACH)
    :param threshold: can be selected with any values
    :param stop_word_type: if customization
    :return: a result file containing generated keywords ("result_GDPR_whole"/"result_REACH_whole")
    """
    root_path = '/'.join(path.split('/')[:-1])

    # load the result file after reference resolution
    if file_type == 'GDPR':
        file_path = os.path.join(root_path, 'output_main/result_file_GDPR.xlsx')
    elif file_type == 'REACH':
        file_path = os.path.join(root_path, 'output_main/result_file_REACH.xlsx')
    else:
        raise Exception('No matched input file ')

    df = pd.read_excel(file_path)

    # Without customization
    if stop_word_type == 0:
        user_word_path = os.path.join(root_path, 'Keyword_Extraction_02/empty_user_keywords.txt')
        stop_word_path = os.path.join(root_path, 'Keyword_Extraction_02/stop_words_nltk.txt')
    else: # with customization
        if file_type == 'GDPR':
            user_word_path = os.path.join(root_path, 'Keyword_Extraction_02/user_keywords_GDPR.txt')
            stop_word_path = os.path.join(root_path, 'Keyword_Extraction_02/stop_words_GDPR.txt')
        elif file_type == 'REACH':
            user_word_path = os.path.join(root_path, 'Keyword_Extraction_02/empty_user_keywords.txt')
            stop_word_path = os.path.join(root_path, 'Keyword_Extraction_02/stop_words_REACH.txt')
        else:
            raise Exception('No matched input file')

    # our extracted keywords, min 1, max 5.
    keywords1 = []
    max_len = 5
    min_len = 1

    # our uncustomized stopwordlist and keyword list
    user_keywords = [w.strip() for w in open(user_word_path, 'r', encoding='utf8').readlines()]
    stop_words = []
    with open(stop_word_path, 'r') as f:
        for w in f.readlines():
            stop_words.append(w.strip())
        f.close()

    # initialize the Rake keyword extractor
    r = Rake(stopwords=stop_words, max_length=max_len, min_length=min_len)

    # loop the whole result file column "gdpr_text" or "reach_text"
    gdpr_reach_texts = []
    if file_type == 'GDPR':
        gdpr_reach_texts = df.gdpr_text.tolist()
    else:
        gdpr_reach_texts = df.reach_text.tolist()
    for i, text in enumerate(gdpr_reach_texts):
        # if without reference_text--> do not extract the keyword
        if str(df.whole_referenced_texts.tolist()[i]) == 'nan':
            phrases = []
            phrases2 = []
        else:
            # extract the user keywords
            user_keywords_in = []
            for w in user_keywords:
                if w.lower() in text.lower():
                    user_keywords_in.append(w.lower())
            # maximal 5 keywords
            if len(user_keywords_in) >= 5:
                keywords1.append(user_keywords_in[:5])
                continue
            else:
                phrases2 = user_keywords_in.copy()

            text = re.sub('[^a-zA-Z]', ' ', text)
            r.extract_keywords_from_text(text)
            # rank the extracted keywords
            phrases = r.get_ranked_phrases_with_scores()
            # leave out the keywords, which their scores are lower than the threshold
            phrases2.extend([p[1] for p in phrases if len(p[1]) > 1 and p[0] > threshold and p[1] not in phrases2])

        if len(phrases2) >= 5:  # maximal 5 keywords
            keywords1.append(phrases2[:5])
        elif 0 < len(phrases2) < 5:  # take the rest
            keywords1.append(phrases2 + [''] * (5 - len(phrases2)))
        else:
            if len(phrases) >= 5:
                keywords1.append(phrases[:5])
            else:
                keywords1.append(phrases + [''] * (5 - len(phrases)))

    # the extracted keywords are saved to a new result file, which is used for our further pipeline
    df2 = pd.DataFrame(keywords1)
    df2.columns = ['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5']
    df3 = pd.concat([df, df2], axis=1)
    df3.to_excel(root_path + '/output_main/result_{}_whole.xlsx'.format(file_type), index=None)
