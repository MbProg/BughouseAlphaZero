# Bughouse Alpha-Zero
<!-- ABOUT THE PROJECT -->
## About The Project
In this project we present a deep learning solution based on AlphaZero for playing ["Bughouse"](https://en.wikipedia.org/wiki/Bughouse_chess) , a chess variant also known as "tandem chess". We use supervised learning in the form of a Deep Convolutional Recurrent Neural Network (DCRNN). The main point of our approach is that we use an asynchronous Monte Carlo tree search algorithm to support the neural network. In addition, a Bughouse server environment is implemented on which real games can be played. Our resulting bughouse engine is then tested and evaluated against other engines in the bughouse environment.

### Bemerkung:

Anbei sind zum einen der Bughouseserver von Moritz Willig. Man muss NodeJS und npm installieren und dann einfach die Schritte im README.md im Ordner folgen. Zum Anderen den Code für die Kommunikation, fündig geworden auf StackOverflow: https://stackoverflow.com/questions/49878953/issues-listening-incoming-messages-in-websocket-client-on-python-3-6

Der Server wird noch weiter verfeinert. Die Git Repo: https://github.com/MoritzWillig/tinyChessServer.git

