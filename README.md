
# COMP3314 project - transformer model reproduction
> We try to reproduce the transformer model in the "attention is all you need"
> paper using tensor2tensor. We train a de-en model using the original parameters
> and data in the original problem for 70k training steps (~15 hours) and 
> the model now obtain a performance of ~23 BLEU (uncased) and ~22 BLEU (cased).

### Steps
1. Setup the python environment: install tensorflow
2. Use `t2t-datagen` to generate the data and the information for the training.`$DATA_DIR` is where the tokenized input is stored and `$TMP_DIR` keeps the raw parallel corpus. Tensor2tensor will download file and tokenize the words automatically if there is nothing in the tmp_dir. `translate_ende_wmt32k` is a problem class in t2t that defines the data (german to english) and config used in the model.
    ```sh
    $ t2t-datagen \
    $ --data_dir=$DATA_DIR \
    $ --tmp_dir=$TMP_DIR \
    $ --problem=translate_ende_wmt32k
    ```
3. Use `t2t-trainer` to train the model. Here we use the basic setting. `$TRAIN_DIR` is where the (intermediate) models and the hyperparameters are stored.
   ```sh
   $ t2t-trainer \ 
   $ --data_dir=$DATA_DIR \
   $ --problem=translate_ende_wmt32k \
   $ --hparams_set=transformer_base \
   $ --output_dir=$TRAIN_DIR
   ```
4. Decode a file: The model is ready to be run after a sufficient training steps. `t2t-decoder` allows both file translation and interactive translation.
   ```sh
   $ t2t-decoder \ 
   $ --data_dir=$DATA_DIR \
   $ --problem=translate_ende_wmt32k \
   $ --hparams_set=transformer_base \
   $ --output_dir=$TRAIN_DIR \
   $ --decode_hparams="beta=4,alpha=0.6" \
   $ (If interactive) --decode_interactive
   $ (else) --decode_from_file=$FROM \
   $        --decode_to_file=$TO
   ```


External source:
[bilinguis.com](http://bilinguis.com)
[The Stanford NLP Group](https://nlp.stanford.edu)

### Result
----

Some translations are listed below. We can observe the model fails to translate some words from the new input (Raupe) and there is difference between the original and the translatied tenses.

| German | English | Translation |
| ------ | ------ | ------ |
| Sie stellte sich also auf die Fußspitzen und guckte über den Rand des Pilzes, und sogleich begegnete ihr Blick dem einer großen blauen Raupe, die mit kreuzweise gelegten Armen da saß und ruhig aus einer großen Huhka rauchte, ohne die geringste Notiz von ihr noch sonst irgend Etwas zu nehmen. | She stretched herself up on tiptoe, and peeped over the edge of the mushroom, and her eyes immediately met those of a large caterpillar, that was sitting on the top with its arms folded, quietly smoking a long hookah, and taking not the smallest notice of her or of anything else. | So she looked at the foothills and looked at the edge of the mushrooms, and so she met the view of a large blue Raupe, who sat with crossroads and smoked out of a large Huhka without taking the slightest note of her or anything else. |
| „Bitte,“ sagte Alice, „wie soll ich denn hineinkommen?“ | 'Please, then,' said Alice, 'how am I to get in?' | “Please,” Alice said, “how should I come into?” |
| Beim Cloud Computing findet eine große Veränderung statt , weil europäische Unternehmensleiter sich nun bewusster sind , dass in den USA gespeicherte Daten der dortigen Rechtsprechung unterliegen und deshalb potenziell gefährdet sind . | A big shift is occurring in cloud computing where European executives have become more aware that data stored in the US is subject to that jurisdiction and therefore potentially vulnerable . | Cloud Computing is a major change, because European corporate leaders are now more confident that data stored in the US is subject to jurisdiction and are potentially at risk. |
| Er lief auf Sand die gesamte Länge der Arena und obwohl er so winzig aussah , hat er phantastische Arbeit geleistet . | He was running on sand down the full length of the arena and even though he looked so tiny he did a marvellous job. | He ran on Sand the entire length of the arena, and although he so wintens aussah , he has done fantastic work. |
| Die Anlage ist mit farbigen LEDs ausgestattet , die so kräftig leuchten , dass die Lichter von den Autofahrern beispielsweise auch bei tiefstehender Sonne gut zu erkennen sind . | The system is fitted with coloured LEDs , which are bright enough that drivers can easily see the lights , even when the sun is low in the sky. | The plant is equipped with coloured LEDs, which are so powerful that the lights of the car drivers are well visible, for example, at the lowest sun. |
| Wer hat die Kuchen gestohlen? | Who Stole the Cakes? | Who has stolen the cakes? |
| Eine Obduktion ergab laut Polizei keine Anhaltspunkte für ein Kapitalverbrechen | According to police , an autopsy showed no evidence of any capital offence . | According to the police, there was no evidence for a capital crime . |
| Ein weiterer , besonders wichtiger Faktor sei die Vernetzung von Hochschulen und Unternehmen . | Another , particularly important factor is that of networking between universities and companies | Another, particularly important factor is the networking of universities and companies . |
| Martin fügte hinzu , dass „ Double Down “ keine definitive Antwort auf die Frage liefere , ob die politischen Sondierungen Obamas Schreibtisch erreicht hätten | Martin added that “Double Down” does not definitively answer whether the political probing reached Obama’s desk . | Martin added that " Double Down " did not provide a definitive answer to the question of whether Obama’s political sonations had reached his desk|
|Im Hauptrennen waren in diesem Jahr noch mehr absolute Topathleten am Start . | This year , there were even more absolute top athletes at the starting line for the main race | In the main race, more absolute top athletes were on the start this year.


### Using own training set
---
Besides using the original data, we also define a problem that uses custom data. The problem definition is in translate_ende_custom.py (ref 1, 2). We use the commoncrawl dataset as the train set and the news-commentary-v13 dataset as the test set to perform en-de translation. The model reaches 100k train steps, however, it performs poorly due to insufficient dataset, it obtains ~4 bleu score in the newstest2014 dataset.



### Todos
---
 - Deploy the model
 - Adjust hyperparamters

### Reference/Further study
---
1. [Tensor2tensor define new problem](https://tensorflow.github.io/tensor2tensor/new_problem.html)
2. [Train a tensor2tensor model using own training set (chinese)](https://blog.csdn.net/hpulfc/article/details/81172498)
3. [Let's build attention is all you need](https://medium.com/datadriveninvestor/lets-build-attention-is-all-you-need-2-2-11d9a29219c4)
4. [Serving tensor2tensor model](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/serving)
