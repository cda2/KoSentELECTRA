# KoSentELECTRA
다양한 구어체 말뭉치들을 모아서 직접 프리트레인, 파인튜닝한 감성 분석 특화 모델입니다.

현재 1개의 모델을 프리트레인을 마쳤으며, 나머지 4개의 모델을 프리트레인 하고 있습니다.

각 모델마다 다른 설정과 말뭉치를 사용하였으며, 단순히 감성 분석에 특화된 모델이 아닌,

다량의 말뭉치를 사용하여 구어체와 문어체 및 신조어, 오탈자까지도 처리할 수 있는 만능 모델을 만들고자 하고 있습니다.

이 모델은 [2020 국어 정보 처리 시스템 경진 대회](http://hkd.or.kr/) 출품작입니다.

## How to use

```python
from transformers import ElectraTokenizer, ElectraModel

tokenizer = ElectraTokenizer.from_pretrained("damien-ir/kosentelectra-discriminator-v2")
model = ElectraModel.from_pretrained("damien-ir/kosentelectra-discriminator-v2")
```

## How to Finetuning / Benchmark
1. 다음의 명령어를 쉘 창에서 입력하여 이 저장소의 파일을 복사합니다.
```shell script
git clone https://github.com/damien-ir/KoSentELECTRA
cd KoSentELECTRA
```

2. 학습용 데이터와 검증용 데이터를 clone한 KoSentELECTRA 폴더에 넣어줍니다.

    기본 설정 상으로 파일의 이름은 NSMC의 파일 이름과 같은 ratings_train.txt, ratings_test.txt이며,
    
    다른 파일 이름을 사용하시려면 config.json 파일을 수정해주세요.
      
```shell script
git clone https://github.com/e9t/nsmc
cp nsmc/ratings_* .
```

3. 파인튜닝 되어있는 모델을 사용하시려면, ```config.json``` 파일 속 ```model_name``` 을 다음과 같이 수정해줍니다.

    파인튜닝 되어있는 모델은 NSMC 외 직접 크롤링 한 데이터를 추가로 학습시킨 모델이므로, 성능의 편차가 클 수 있습니다.
```json
{
  "model_name": "damien-ir/kosentelectra-discriminator-v2-finetuned"
}
```

4. docker의 --gpus all 명령어를 사용할 수 있다면, 다음 명령어를 실행해서 바로 이진 클래스 분류를 실행할 수 있습니다.

    --rm 옵션을 사용 시 컨테이너가 종료될 때 자동으로 삭제됩니다.

```shell script
docker run --rm --gpus all -v $(pwd):/base-dir damienir/hkd-electra:v2-finetuned-benchmark
```

5. 이후 모델 학습 결과가 KoSentELECTRA 폴더 속에 자동적으로 저장됩니다.

* docker를 이용한 학습이 싫으시다면, 직접 ```classification.py``` 를 실행하여 fine-tuning / benchmark를 실행할 수 있습니다.

* Windows 10의 환경이라 GPU 문제로 인해 docker 설정이 어려운 경우,

    nvidia의 [cuda toolkit documentation](https://docs.nvidia.com/cuda/wsl-user-guide/index.html), simpletransformers의 [setup](https://github.com/ThilinaRajapakse/simpletransformers#setup) 을 참고하여 GPU 환경을 구축 후 실행해 주세요.

* wandb를 사용하고자 하는 분은 config.json 파일에 ```"wandb_project": "wandb 프로젝트 이름"``` 을 넣으시면 잘 작동합니다.

* config 설정은 simpletransformers의 [Args Explained](https://github.com/ThilinaRajapakse/simpletransformers#args-explained) 를 참고해주세요.


도커가 정상적으로 실행되었다면 다음과 같이 모델이 학습되는 모습을 볼 수 있습니다.
![docker works well](https://raw.githubusercontent.com/Damien-IR/KoSentELECTRA/master/images/docker_works_well.png)
    
+) 다음의 명령어를 사용하여 간단한 감성 분석 서버를 만들 수 있습니다. localhost:8000으로 접속해보세요.

```shell script
docker run --rm --gpus all -p 8000:8000 --name nsmc-web damienir/hkd-electra:nsmc-web
```

![nsmc_web_image](https://raw.githubusercontent.com/Damien-IR/KoSentELECTRA/master/images/nsmc_web.gif)

## About Model's Corpus
개인으로서 수집할 수 있는 대용량 말뭉치는 다 사용하였으며, 각 모델마다 차이가 있습니다.

사용한 코퍼스 목록들은 다음과 같습니다.
* [KcBERT Corpus](https://www.kaggle.com/junbumlee/kcbert-pretraining-corpus-korean-news-comments)
* [Wikipedia](https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4_%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C)
* [NSMC](https://github.com/e9t/nsmc)
* [KCC](http://nlp.kookmin.ac.kr/kcc/)
* [국어국립원 모두의 말뭉치](https://corpus.korean.go.kr/) 구어 / 문어 / 신문 / 메신저 / 웹 말뭉치
* 직접 크롤링한 음식점 리뷰 (주로 구어체, 21GB)


모든 vocab는 tokeniers의 BertWordPieceTokenizer를 사용하여 만들었으며,

limit_alphabet을 **1만 이상**으로 설정하여 생성하였습니다.

각 모델 별 Vocab와 말뭉치의 차이는 다음과 같습니다.

* Model 1
    * Vocab size 32000, KcBERT + 음식점 리뷰 + NSMC + 위키피디아 + KCC 사용
* Model 2
    * Vocab size 32000, KcBERT + 음식점 리뷰 (소량) + NSMC 사용, 1M steps 학습
* Model 3
    * Vocab size 64000, KcBERT + 음식점 리뷰 + NSMC + 위키피디아 + KCC + **국립국어원 모두의 말뭉치**
* Model 4  
    * Vocab size 128000, KcBERT + 음식점 리뷰 + NSMC + 위키피디아 + KCC + **국립국어원 모두의 말뭉치**
* Model 5 (for KorQuad testing)
    * Vocab size 32000, 인라이플의 [Large 모델](https://github.com/enlipleai/kor_pretrain_LM) vocab 사용, 말뭉치는 위와 같음
    
모델 2를 제외한 나머지 모델들은 아직도 TPU에서 학습 중이므로, 추후 결과를 알려 드리겠습니다.

## About Pretrain Config
구어체 기반의 모델을 학습시에는 다소 Base 모델의 기본 학습률인 2e-4가 높다고 판단하였고,

이를 1.5e-4 로 수정하여 학습시켰습니다.

```json
{
    "use_tpu": true,
    "num_tpu_cores": 8,
    "tpu_name": "your-tpu",
    "tpu_zone": "europe-west4-a",
    "model_size": "base",
    "vocab_size": 32000,
    "do_lower_case": false,
    "learning_rate": 1.5e-4,
    "keep_checkpoint_max": 10,
    "train_batch_size": 256,
    "save_checkpoints_steps": 20000,
    "num_train_steps": 1300000
}
```

## About Finetuning
[simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers) 를 이용한 파인 튜닝 코드를 작성하였으며,

내용은 이 리포지토리에 첨부 되어있는 classification.py 파일과 별반 차이 없으며, 경우에 따라 학습률이나 세부 파라미터를 조정하는 정도입니다.

최종 업로드 모델은 크게 파라미터를 손 볼 필요가 없어 쉽게 재현이 가능한 5e-5의 lr을 설정한 모델을 업로드 하였습니다.

기존 파인튜닝 코드들의 경우 성능을 비교하거나, 한 눈에 보기 위해서는 다소 시간을 소요해야 했으나,

simpletransformers와 그에 내장되어 있는 [wandb](https://app.wandb.ai) 를 사용하여 빠르고, 한 눈에 성능을 비교할 수 있습니다.

## Benchmark Result
배치 사이즈, 학습률 등의 설정을 조정하여 NSMC 태스크에서 최고 정확도를 91.49%까지 달성하였습니다.

S3에 업로드 되어 있는 [모델](https://huggingface.co/damien-ir/kosentelectra-discriminator-v2) 의 성능은 91.35%, lr은 5e-5 이므로 해당 모델 사용에 오해 없으시길 바랍니다.

해당 성능 측정은 [Simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers) 와 [wandb](https://app.wandb.ai) 를 사용하여 측정하였습니다.

![Benchmark Result](https://raw.githubusercontent.com/Damien-IR/KoSentELECTRA/master/images/benchmark_result.png)


## Acknowledgement
TensorFlow Research Cloud(TFRC) 의 지원을 받아 Cloud TPU로 모델을 학습하였습니다.<br>

## Reference
- [ELECTRA](https://github.com/google-research/electra)
- [KcBERT](https://github.com/Beomi/KcBERT)
- [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)
- [kor_pretrain_LM](https://github.com/enlipleai/kor_pretrain_LM)
- [wandb](https://app.wandb.ai)
