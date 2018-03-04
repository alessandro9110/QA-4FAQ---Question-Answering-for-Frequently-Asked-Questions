# QA-4FAQ---Question-Answering-for-Frequently-Asked-Questions
Web site: http://qa4faq.github.io/#ref-1

Trovare FAQ su un sito web è un task critico: gli utenti che navigano nella FAQ page, potrebbero essere scoraggiati nel non ricevere la giusta risposta ad una precisa domanda. 

Il seguente task proposto, consiste nel restituire una lista di rilevanti FAQs in corrispondenza di una precisa domanda, data una lista di risposte. Il task consiste nella realizzazione di una rete neurale con Keras.

Nello specifico, la rete prende due input, le questions e le answers e restituisce uno score di similarità. Più la similarità tende ad 1, più la rispsota è correlata alla domanda. Sono stati realizzati due modelli:

##### Model 1: Conv2D - MaxPooling (Conv2D - MaxPooling)
Prende in input questions e answers, i quali input vengono trasformati in Embedding e passati al Bi-LSTM. Successivamente viene realizzato un prodotto di matrici tra le matrici di Embedding e Bi-LSTM. I prodotti vengono concatenati in un'unica matrice, la quale viene passata al Convutional2D e successivamente al Maxpooling.
Infine il risultato viene passato ad un MLP.
Il modello restituisce un valore di score che indica la similarità tra domanda e risposta.


##### Model 2: Average BiLSTM - Embedding (AVG Embedding - Bi-LSTM)
Prende in input questions e answers, i quali input vengono trasformati in Embedding e passati al Bi-LSTM.


### TokW2V.ipynb
Fase di preprocessing nella quale vengono "tokenizzate" questions, answers e test questions. Il tokenizzatore utilizzato è stato fornito dal professore Giuseppe Attardi, ed è disponibile al seguente indirizzo web http://tanl.di.unipi.it/it/overview.html . Dopo aver tokenizzato le sentences, viene utilizzato il Word2Vec, algoritmo  che richiede in ingresso un corpus e restituisce un insieme di vettori che rappresentano la distribuzione semantica delle parole nel testo e realizza un dizionario di termini.





#### Armillotta Alessandro a.armillotta91@gmail.com
#### Inversi Alessandro a.inversi@gmail.com 
#### Savasta Davide d.savasta592@gmail.com 
