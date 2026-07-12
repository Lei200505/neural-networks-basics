# Neural-networks-basics

A repóban szereplő példákhoz egyes helyi adathalmazok szükségesek.

## Linear Regression from scratch - linear_regression.py

Lineáris regressziós modellt valósít meg saját implementációval. A modell gradiens ereszkedés-descet segítségével tanulja meg a feltételezett paramétereket egy zajos adathalmaz alapján. Emellett vizualizálja a regreszziós folyamatot.

## Logistic Regression from scratch - logistic_regression.py

Logisztikus regressziós modellt valósít meg saját implementációval. Kétdimenziós mesterséges/véletlen adathalmazon tanít egy sigmoid aktivációs függvény, log loss veszteségfüggvény és gradiens alapú frissítés segítségével.

## Softmax Regression from scratch - softmax_regression.py

Többosztályos logisztikus regresszió modell saját implementációval. A modell softmax aktivációs függvényt használ, one-hot elkódolással és cross-entropy lossal tanítja be a modellt, gradiens alapú javítólépésekkel.

## MNIST with Softmax Regression - mnist_softmax_regression.py

MNIST adathalmazon alkalmazott többosztályos softmax regressziós modellt valósít meg. Kezírással történt 10 számjegy felismerését végzi el az implementált softmax regresszió segítségével.

## Adam optimizer from scratch - adam_optim.py

Adam optimizer egy implementációja

## CIFAR-10 Classification with Neural Networks - MLP_Convnet.py

A program egy egyszerű neurális hálót (MLP) és egy konvolúciós neurális hálót (CNN) tanít a CIFAR-10 képadathalmazon PyTorch segítségével. Betölti és normalizálja az adatokat, létrehozza a modelleket, betanítja SGD optimalizálóval, majd kiértékeli a teszt adathalmazon a pontosságukat.

## Auto MPG Regression with Neural Networks - auto_mpg.py

Többrétegű neurális háló segítségével (MLP) tanít be egy modellt az Auto MPG adathalmazon, különböző járművek üzemanyag-fogyasztásának becslésére. Adatfeldolgozás (tisztítás, standardizálás, one-hot-encoding), a modell ReLu aktivációs függvényt használ ADAM optimalizálóval és MSE loss-függvénnyel, validációs adathalmazzal.

## AlexNet on Cifar10 with 83% accuracy - alexnet.py

AlexNet architektúrájú neurális hálót valósít meg a CIFAR-10 képadathalmazon (10 képosztály)
Adatfeldolgozás (normalizálás, batchek, adataugmentáció).


## MLP Autoencoder MNIST rekonstrukció - autoencoder.py

MLP autoencoder implementáció az MNIST adathalmazon. Encoder és decoder részből áll, 28*28 pixeles képet egy 32 dimenziós látens térbe tömöríti, majd a decoder ebből alkotja újra a képet. ReLu és sigmoid aktiváció, Adam optimizer és BCE lossfüggvény.

## Convolutional Autoencoder MNIST - conv_autencoder.py

Konvolúciós autoenkóder implementáció. Az MLP autoenkóderrel szemben megtartja a kép szerkezetét és kihasználja azt ezzel plusz információt elraktározva a képről. ReLu és sigmoid aktivációs függvény, Adam optimizer, MSE lossfüggvény. 

## Transfer Learning ResNet18 képosztályozás - transfer_learning.py

Egy előre betanított ResNet18 konvolúciós háló segítségével osztályozza méheket és egyéb rovarokat. 
Az első betanítés során a teljes ResNet18 hálózat finomhangolása történik. Az ImageNet adathalmazon előre betanított modellt használja, majd az utolsó teljesen összekötött réteget lecseréli a két célosztály felismerésére. A hálózat minden paramétere frissül a tanítás során.
A második betanítás során a konvolúciós rétegek súlyai rögzítve vannak, így csak az újonnan hozzáadott osztályozó réteg tanul. Ez lehetővé teszi a már megtanult vizuális jellemzők új feladatra való felhasználását kisebb adathalmaz esetén is.

## RNN - rnn.py

Rekurrens neuráis háló implementációja, személynevek alapján próbálja meghatározni azok eredetét. Karakterizációt
hajt végre, karakterminntákat tanul. One-hot-encoding, rekurrens réteg, logsoftmax (valószínűségi kategóriák), negatív-loglikelihood.

