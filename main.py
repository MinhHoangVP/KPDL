from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

f = open("dataresult.txt", "r")
result = f.readlines()
f2 = open("datatrain.txt", "r")
corpus2 = f2.readlines()
f3 = open("datatest.txt", "r")
test = f3.readlines()

vectorizer = CountVectorizer()
vectorizer2 = CountVectorizer()
vectorizer3 = CountVectorizer()

traindata = vectorizer2.fit_transform(corpus2)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(traindata, result)

rsarray = []
count = 1
for xnn in test:
    xnn2 = ['']
    xnn2[0] = xnn
    xnn2.extend(vectorizer2.get_feature_names())
    testdata = vectorizer3.fit_transform(xnn2)[0]
    if testdata.shape[1] != traindata.shape[1]:
        xnn3 = ['']
        for x in xnn.split(' '):
            check = 0
            for y in vectorizer2.get_feature_names():
                if x == y:
                    check = 1
                    break
            if check == 1:
                xnn3[0] = xnn3[0] + x + ' '
        # tao vector test
        xnn3.extend(vectorizer2.get_feature_names())
        testdata = vectorizer3.fit_transform(xnn3)[0]
    rs = neigh.predict(testdata)
    print (rs[0])
    rsarray.append(rs[0])
    print (count)
    count += 1

w3 = open("testrs.txt", 'w+')
for x in rsarray:
    w3.write(x)
w3.close()
print ('done!')