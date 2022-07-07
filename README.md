<h1 align="center"><b>Part #1. Mbart NMT Engine for LC(Letter of Credit) </b></h1>
This project is about building a NMT ENGINE using MBARTCC25 PLM for LC type's documentation(EN > KO).

## 🚀 Features
- Possible to use custom sentencepiece model & vocab
- Reduce token embedding matched custom spc vocab and convert fairseq type's model to huggingface type
- Good working on small domain data(<1M) and different language family
- Futher performance benifit using back stranslation augmentation
- Transformer architecture(bidirectional encoder and autoregressive decoder)

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

### 4. Reduce fairseq's mbartcc25 plm and convert to huggingface format
```
(mbart_env)$ python reduce_fairseq_plm.py
```

### 5. Finetune
```
(mbart_env)$ cd scripts
(mbart_env)$ ./finetune.sh
#need to edit finetunining info
```

### 6. evaluate
```
(mbart_env)$ cd scripts
(mbart_env)$ ./evaluate.sh
#need to edit evaluation info
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

This project is about building a web application to translate LC type's documentation(EN > KO) using Mbart translator API. Here you will be able to translate between EN > KO languages. Whenever you type something, it will be automatically translated in the side panel. 


<br/>

![Language Translator App](https://user-images.githubusercontent.com/60684500/177440422-1c50494c-b261-4735-98f7-08a1f728daaf.png)
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
cd /home/workspace/mbart-nmt/Translator
```


## Install

Install NPM
To check if you have Node.js installed, run this command in your terminal:

```
$ curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
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
