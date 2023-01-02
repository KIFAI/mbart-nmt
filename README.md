<h1 align="center"><b>Part #1. Mbart NMT Engine </b></h1>
This project is about building a NMT Engine with powerful translation performance from using mBart architecture 

- ****Multilingual Denoising Pre-training for Neural Machine Translation(mBart25)****
    - [https://arxiv.org/pdf/2001.08210.pdf](https://arxiv.org/pdf/2001.08210.pdf)
- **Multilingual Translation with Extensible Multilingual Pretraining and
Finetuning(mBart50)**
    - [https://arxiv.org/pdf/2008.00401.pdf](https://arxiv.org/pdf/2008.00401.pdf)
    
## 🚀 Features
- **Possible to use custom sentencepiece model & vocab**
    - you can train own sentencepiece model & vocab or use pretrained sentencepiece model & vocab
- **Reduce token embedding size to fit custom spc vocab and convert it huggingface type model(mbart-m2m)**
    - We used offical “facebook/mbart-large-50-many-to-many-mmt”(multilingual mBart fintuned model), which can be downloaded from Huggingface. And we customized its token embedding&lm head since we don’t need another languges except for korean & english. In addition, we were confident in the composition of the vocab in Korean that could produce good performance. so we tried to use the self-made sentence model and vocab as much as possible.
    - In fact, the vocab of the official mBart50-m2m model, which is about 250,000 in size, influenced the inference latency and memory issues during training, and we conducted the training with only the necessary "Korean and English." In the case of Token, which was not in the vocab of the Hugging Face model, it was updated while finetuning and had little impact on performance degradation.
- **Good working on small domain data(<1M) and different language family**
    - When comparing the performance of the same parallel sentence set of “en > ko and ko to en”, the performance difference tends to be very large, but as a bidirectional model, we significantly reduced the performance difference compared to other translators (Google/Papago/T5, etc.)
- **Futher performance benifit using data augmentation(integrate sentences into segments unit by similar domain)**
    - We've collected about 10 million pairs of parallel corporations and doubled them by merging in random order(random permutation) to form a phrase
- **Transformer architecture(bidirectional encoder and autoregressive decoder)**
    - When you use Transformer(Encoder & Decoder), which called standard seq2seq arhictecture, it can be speed up by using ctranslate2
    - Although NAT(Non-Autoregressive Transformer) research is actively underway in the field of machine translation, it is still known that the Autoregressive transformer model has strong performance in the Generation and Translation Task yet.
- **Bidirectional translation engine(EN <> KO)**
    - When learning a model, it takes twice as long because the directions of the original and target language sentences are switched to each other, but in the model deployment environment, it uses less computing resources and enables two-way translation
- **Possibile using fast model convertor from using Ctrans2(CPU & GPU) or Onnx(only CPU ver)**

### 1. Dependency
```
$ python3 -m venv mbart_env
$ . mbart_env/bin/activate
(mbart_env)$ pip install --upgrade pip
(mbart_env)$ pip install -r requirements.txt
```

### 2. Monolingual Corpus
```
(mbart_env)$ cd src/raw_data
(mbart_env)$ cat train.* > ../sentencepiece/monolingual_corpus.txt
```

### 3. Sentencepiece Model & Vocab
```
(mbart_env)$ python prepare_sentencepiece
```

### 4. Reduce huggingface's mbart50-m2m
```
(mbart_env)$ python reduce_hf_plm.py --plm_name facebook/mbart-large-50-many-to-many-mmt --plm_local_path ./src/plm/tmp_ckpt --reduction_path ./src/plm/reduced_hf_mbart50_m2m
```

### 5. Finetune
```
(mbart_env)$ cd scripts
(mbart_env)$ ./finetune.sh
#need to edit finetunining info
```

### 6. evaluate
```
(mbart_env)$ python evaluate.py
```

<h1 align="center"><b>Part #2. Translator App</b></h1>

<h2 align="center">A web application to translate multiple languages</h2>    

<br />

<p align="center">
    <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="reactjs" />
    <img src="https://img.shields.io/badge/JavaScript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E" alt="javascript"/>
    <img src="https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white" alt="html5"/>
    <img src="https://img.shields.io/badge/Rest_API-02303A?style=for-the-badge&logo=react-router&logoColor=white" alt="restAPI"/>
    <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="css3"/>     
</p>
    
<br/>

This project is about building a web application to translate sentence and documentation(EN <> KO) using Mbart translator API. Here you will be able to translate between EN <-> KO languages. Whenever you type something, it will be automatically translated in the side panel. 


<br/>

![image](https://user-images.githubusercontent.com/60684500/195746993-98d53da2-ec47-4dad-b243-83e94e3c8d24.png)

<br/>


## 🚀 Features
- Translate automatically without clicking on any button
- Debounce is used to avoid sending unnecessary network requests
- Copy to clipboard feature available 
- Word count feature available 
- Clear all text at one click feature available 
- Bubble animations
- Responsive for all screen sizes


### Prerequisites

- NPM 

### Setup


The project repository can be found in https://github.com/jyoyogo/mbart-nmt.git or just clone the project using this command. 


```
Using HTTPS

# git clone  https://github.com/jyoyogo/mbart-nmt.git
```

+ Open terminal on your workspace with

```
cd /pwd/WebApp/Translator
```


## Install

Install NPM
To check if you have Node.js installed, run this command in your terminal:

```
$ curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
$ sudo apt-get install -y nodejs


$ node -v
v14.19.
$ npm version
{
  'translation-app': '0.0.0',
  npm: '6.14.16',
  ares: '1.18.1',
  brotli: '1.0.9',
  cldr: '37.0',
  http_parser: '2.9.4',
  icu: '67.1',
  llhttp: '2.1.4',
  modules: '72',
  napi: '8',
  nghttp2: '1.41.0',
  node: '12.22.12',
  openssl: '1.1.1n',
  tz: '2021a4',
  unicode: '13.0',
  uv: '1.40.0',
  v8: '7.8.279.23-node.57',
  zlib: '1.2.11'
}
```

To install all the dependences of the project, run the following command:

```
npm install
```

To run the application, run the following command:

```
npm run dev -- --port ${PORT_NUM} --host=${HOST}
```

<h1 align="center"><b>Part #3. Translator Bot</b></h1>

<h2 align="center">Chat application to translate languages interacively</h2>    

<br />

<img width="1279" alt="image" src="https://user-images.githubusercontent.com/60684500/202968404-9a4f2412-c2ea-4c00-a7ef-5e3bab9913bd.png">

<br/>


## 🚀 Features
- Translate automatically english to korean or korean to english
- Write translation history in enko or koen channel if you type '저장하기' 
- Use mattermost App

### Prerequisites

- docker
```
$ docker pull mattermost/mattermost-preview
$ docker run -it -d --name mattermost-preview -p 8065:8065 mattermost/mattermost-preview:latest
```

### Setup


This project should be preceded by the 'build_docker README.md' guide procedure. 

And you need to edit "url info and trans bot token" in mm_transbot.py

```
$ cd /pwd/build_docker
```

### Run TransBot
If you complete setting docker container, type below command

```
$ python mm_transbot.py --url ${URL} --mm_port ${MM_PORT} --trans_port ${TRANS_PORT} \
            --bot_access_token ${BOT_ACCESS_TOKEN} --login_id ${LOGIN_ID} --password ${PASSWORD}
```

<h1 align="center"><b>Part #4. MQM Viewer </b></h1>
For more details about the MQM Viewer web app, refer to the corresponding link.(https://github.com/google-research/google-research/tree/master/mqm_viewer)
<h2 align="center">Viewer application for Multidimentional Quality Metric</h2>    

<br />

<img width="1274" alt="image" src="https://user-images.githubusercontent.com/60684500/210199798-00973222-cd27-4bfe-81ee-46db1826e2db.png">
<img width="1279" alt="image" src="https://user-images.githubusercontent.com/60684500/210199947-770a94cb-cff3-49e0-9e08-2f42e8a68a36.png">
<img width="1273" alt="image" src="https://user-images.githubusercontent.com/60684500/210200489-02af4dbe-8c48-4823-b6a9-14ef8b888988.png">


<br />

```
$ cd /pwd/WebApp/MQM_Viewer
$ http-server -p $PORT -c 1 #No Caching Times
```
