### NER
Tugas NLP-nya George, Zaky, dan Thoyib.

Just do:
```sh
$ git clone https://github.com/grgp/NLP-NERTweets.git
```

Then run:
```sh
$ python main.py
```

By default, the code used the dump data (pickles/*.pickle) to avoid always preprocessing the training data. To reload the data:
```sh
$ python main.py reload
```

### Licenses
```
ner-indonesia
@mastersthesis{ Syaifudin2016,
       author       = "Yusuf Syaifudin",
       title        = "QUOTATIONS IDENTIFICATION FROM INDONESIAN ONLINE NEWS USING RULE-BASED METHOD",
       school       = {Universitas Gadjah Mada},
       note         = {Undergraduate Thesis}
       year         = "2016" }

@mastersthesis{ Fachri2014,
       author       = "Muhammad Fachri",
       title        = "NAMED ENTITY RECOGNITION FOR INDONESIAN TEXT USING HIDDEN MARKOV MODEL",
       school       = {Universitas Gadjah Mada},
       note         = {Undergraduate Thesis}
       year         = "2014" }
```