# neural-networks-basics


# Linear Regression from scratch - linear_regression.py

Lineáris regressziós modellt valósít meg saját implementációval. A modell gradiens ereszkedés-descet segítségével tanulja meg a feltételezett paramétereket egy zajos adathalmaz alapján. Emellett vizualizálja a regreszziós folyamatot.

# Logistic Regression from scratch - logistic_regression.py

Logisztikus regressziós modellt valósít meg saját implementációval. Kétdimenziós mesterséges/véletlen adathalmazon tanít egy sigmoid aktivációs függvény, log loss veszteségfüggvény és gradiens alapú frissítés segítségével.

# Softmax Regression from scratch - softmax_regression.py

Többosztályos logisztikus regresszió modell saját implementációval. A modell softmax aktivációs függvényt használ, one-hot elkódolással és cross-entropy lossal tanítja be a modellt, gradiens alapú javítólépésekkel.

# MNIST with Softmax Regression - mnist_softmax_regression.py

MNIST adathalmazon alkalmazott többosztályos softmax regressziós modellt valósít meg. Kezírással történt 10 számjegy felismerését végzi el az implementált softmax regresszió segítségével.

# CIFAR-10 Classification with Neural Networks - MLP_Convnet.py

A program egy egyszerű neurális hálót (MLP) és egy konvolúciós neurális hálót (CNN) tanít a CIFAR-10 képadathalmazon PyTorch segítségével. Betölti és normalizálja az adatokat, létrehozza a modelleket, betanítja SGD optimalizálóval, majd kiértékeli a teszt adathalmazon a pontosságukat.

# Auto MPG Regression with Neural Networks - auto_mpg.py

Többrétegű neurális háló segítségével (MLP) tanít be egy modellt az Auto MPG adathalmazon, különböző járművek üzemanyag-fogyasztásának becslésére. Adatfeldolgozás (tisztítás, standardizálás, one-hot-encoding), a modell ReLu aktivációs függvényt használ ADAM optimalizálóval és MSE loss-függvénnyel, validációs adathalmazzal.