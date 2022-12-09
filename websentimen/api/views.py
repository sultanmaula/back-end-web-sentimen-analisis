from django.shortcuts import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import api_view

# machine learning
import pandas as pd
import nltk 
import string
import re

from .serializers import DataSerializer, PreprocessingSerializer, KNNSerializer, SmoteKnnSerializer
from .models import Data, Preprocessing, Kaenen, SmoteKnn

# Create your views here.
@api_view(['POST'])
def preprocessing_post(request):
    serializer = DataSerializer(data=request.data)
    if serializer.is_valid(raise_exception=True):
        serializer.save()
        return Response(serializer.data)
    else:
        return Response(serializer.errors)

@api_view(['GET'])
def preprocessing(request):
    items = Data.objects.all().order_by('id').values()
    df = pd.DataFrame(items)

    #===== CASE FOLDING
    df['content'] = df['content'].str.lower()

    #===== TOKENIZING
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize 
    from nltk.probability import FreqDist

    def remove_tweet_special(text):
        # remove tab, new line, ans back slice
        text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
        # remove non ASCII (emoticon, chinese word, .etc)
        text = text.encode('ascii', 'replace').decode('ascii')
        # remove mention, link, hashtag
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
        # remove incomplete URL
        return text.replace("http://", " ").replace("https://", " ")
                    
    df['content'] = df['content'].apply(remove_tweet_special)
    #remove number
    def remove_number(text):
        return  re.sub(r"\d+", "", text)
    df['content'] = df['content'].apply(remove_number)

    #remove punctuation
    def remove_punctuation(text):
        return text.translate(str.maketrans("","",string.punctuation))
    df['content'] = df['content'].apply(remove_punctuation)

    #remove whitespace leading & trailing
    def remove_whitespace_LT(text):
        return text.strip()
    df['content'] = df['content'].apply(remove_whitespace_LT)

    #remove multiple whitespace into single whitespace
    def remove_whitespace_multiple(text):
        return re.sub('\s+',' ',text)
    df['content'] = df['content'].apply(remove_whitespace_multiple)

    #remove single char
    def remove_single_char(text):
        return re.sub(r"\b[a-zA-Z]\b", "", text)
    df['content'] = df['content'].apply(remove_single_char)

    # NLTK word rokenize 
    def word_tokenize_wrapper(text):
        return word_tokenize(text)
    df['review_tokens'] = df['content'].apply(word_tokenize_wrapper)
    
    def FreqDist_wrapper(text):
        return FreqDist(text)

    df['review_tokens_fdist'] = df['review_tokens'].apply(FreqDist_wrapper)
    df['review_tokens_fdist'] = df['review_tokens_fdist'].head().apply(lambda x : x.most_common())

    #===== FILTERING (STOPWORD REMOVAL)
    nltk.download('stopwords')

    from nltk.corpus import stopwords
    #get stopwords Indonesia
    list_stopwords = stopwords.words('indonesian')

    # ---------------------------- manualy add stopword  ------------------------------------
    # append additional stopword
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                        'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                        'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                        'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                        'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                        'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                        '&amp', 'yah'])

    # ----------------------- add stopword from txt file ------------------------------------
    # read txt stopword using pandas
    txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)

    # convert stopword string to list & append additional stopword
    list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

    # ---------------------------------------------------------------------------------------

    # convert list to dictionary
    list_stopwords = set(list_stopwords)

    #remove stopword pada list token
    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

    df['review_tokens_WSW'] = df['review_tokens'].apply(stopwords_removal)

    #===== NORMALIZATION
    normalizad_word = pd.read_csv("slang_words.txt")

    normalizad_word_dict = {}

    for index, row in normalizad_word.iterrows():
        if row[0] not in normalizad_word_dict:
            normalizad_word_dict[row[0]] = row[1] 

    def normalized_term(document):
        return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

    df['review_normalized'] = df['review_tokens_WSW'].apply(normalized_term)

    #===== STEMMER
    # import Sastrawi package
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    import swifter

    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # stemmed
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in df['review_normalized']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)


    # apply stemmed term to dataframe
    def get_stemmed_term(document):
        return [term_dict[term] for term in document]

    df['review_tokens_stemmed'] = df['review_normalized'].swifter.apply(get_stemmed_term)
    return HttpResponse(df.to_json())
    
@api_view(['GET'])
def preprocessing_reset(request):
    Data.objects.all().delete()
    return JsonResponse({'status': 200, 'message': 'successfully deleted database!'})


#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================

@api_view(['POST'])
def clustering_post(request):
    serializer = PreprocessingSerializer(data=request.data)
    if serializer.is_valid(raise_exception=True):
        serializer.save()
        return Response(serializer.data)
    else:
        return Response(serializer.errors)
    
@api_view(['GET'])
def clustering(request):
    items = Preprocessing.objects.all().order_by('id').values()
    df = pd.DataFrame(items)
    
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    vectorizer = CountVectorizer()
    tf_transformer = TfidfTransformer()

    datavector = vectorizer.fit_transform(df['review_tokens_stemmed'])
    datatfidf = tf_transformer.fit_transform(datavector)
    
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(datatfidf)
    hasil = kmeans.labels_
    df['Cluster'] = hasil
    
    return HttpResponse(df.to_json())

@api_view(['GET'])
def clustering_reset(request):
    Preprocessing.objects.all().delete()
    return JsonResponse({'status': 200, 'message': 'successfully deleted database!'})


#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================


@api_view(['POST'])
def knn_post(request):
    serializer = KNNSerializer(data=request.data)
    if serializer.is_valid(raise_exception=True):
        serializer.save()
        return Response(serializer.data)
    else:
        return Response(serializer.errors)
    
@api_view(['GET'])
def knn(request):
    items = Kaenen.objects.all().order_by('id').values()
    df = pd.DataFrame(items)
    
    content = df['content']
    cluster = df['cluster']
    
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    le = preprocessing.LabelEncoder()
    content_encoded = le.fit_transform(content)
    cluster_encoded = le.fit_transform(cluster)
    features = list(zip(content_encoded, cluster_encoded))

    # Splitting train : test to 70 : 30 ratio
    X_train, X_test, y_train, y_test = train_test_split(features, cluster_encoded, test_size=0.3)

    # Applying k = 3, default Minkowski distance metrics
    model = KNeighborsClassifier(n_neighbors=3)

    # Training the classifier
    model.fit(X_train, y_train)

    # Testing the classifier
    y_pred = model.predict(X_test)
    print('Predicted', y_pred)
    print('Actual data', y_test)

    # confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    accuracy = accuracy_score(y_test, y_pred)
    
    return JsonResponse({
        'status': 200,
        'message': 'success',
        'data': {
            'classification_report': df_report.to_json(),
            'accuracy': accuracy,
        }
    })

@api_view(['GET'])
def knn_reset(request):
    Kaenen.objects.all().delete()
    return JsonResponse({'status': 200, 'message': 'successfully deleted database!'})


#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================


@api_view(['POST'])
def smoteknn_post(request):
    serializer = SmoteKnnSerializer(data=request.data)
    if serializer.is_valid(raise_exception=True):
        serializer.save()
        return Response(serializer.data)
    else:
        return Response(serializer.errors)
    
@api_view(['GET'])
def smoteknn(request):
    items = SmoteKnn.objects.all().order_by('id').values()
    df = pd.DataFrame(items)
    
    content = df['content']
    cluster = df['cluster']
    
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from imblearn.over_sampling import SMOTE
    from collections import Counter

    le = preprocessing.LabelEncoder()
    content_encoded = le.fit_transform(content)
    cluster_encoded = le.fit_transform(cluster)
    features = list(zip(content_encoded, cluster_encoded))
    
    # Splitting train : test to 70 : 30 ratio
    X_train, X_test, y_train, y_test = train_test_split(features, cluster_encoded, test_size=0.3)

    # Oversampling the train dataset using SMOTE
    smt = SMOTE()
    X_train_smote, y_train_smote = smt.fit_resample(X_train, y_train)
    
    # Applying k = 3, default Minkowski distance metrics
    model = KNeighborsClassifier(n_neighbors=3)
    
    # Training the classifier
    model.fit(X_train_smote, y_train_smote)
    
    # Testing the classifier
    y_pred = model.predict(X_train_smote)
    
    # confusion = confusion_matrix(y_train_smote, y_pred)
    report = classification_report(y_train_smote, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    accuracy = accuracy_score(y_train_smote, y_pred)
    
    return JsonResponse({
        'status': 200,
        'message': 'success',
        'data': {
            'classification_report': df_report.to_json(),
            'accuracy': accuracy,
        }
    })
    
@api_view(['GET'])
def smoteknn_reset(request):
    SmoteKnn.objects.all().delete()
    return JsonResponse({'status': 200, 'message': 'successfully deleted database!'})