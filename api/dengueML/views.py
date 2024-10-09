import numpy as np, pickle
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split

@api_view()
def doPrediction(request):
    dataFromPage = np.array([
        request.query_params.get('p1'),
        request.query_params.get('p2'),
        request.query_params.get('p3'),
        request.query_params.get('p4')]).astype(np.float32).reshape((1,-1))
    with open('model.pkl','rb') as f:
        gn = pickle.load(f)
    pred = gn.predict(dataFromPage)
    return Response({'result' : pred[0]})

@api_view()
def trainModel(request):
    iris = datasets.load_iris()
    gn = GaussianNB()
    gn.fit(iris['data'],iris['target'])
    with open('model.pkl','wb') as f:
        pickle.dump(gn,f)        

    return Response({'status' : 'treinamentoRealizado'})
