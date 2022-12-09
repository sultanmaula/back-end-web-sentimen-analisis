from rest_framework import serializers

from api.models import Data, Preprocessing, Kaenen, SmoteKnn

class DataSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Data
        fields = ('username', 'content')

class PreprocessingSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Preprocessing
        fields = ('username', 'content', 'review_tokens_stemmed')

class KNNSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Kaenen
        fields = ('username', 'content', 'cluster')
        
class SmoteKnnSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = SmoteKnn
        fields = ('username', 'content', 'cluster')