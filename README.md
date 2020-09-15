# KoSentELECTRA
다양한 구어체 말뭉치들을 모아서 직접 프리트레인, 파인튜닝한 감성 분석 특화 모델입니다.

현재 1개의 모델을 프리트레인을 마쳤으며, 나머지 4개의 모델을 프리트레인 하고 있습니다.

각 모델마다 다른 설정과 말뭉치를 사용하였으며,

small 모델의 경우 base 모델의 학습이 모두 끝난 후 생성하고자 합니다.

## How to use
```python
from transformers import ElectraTokenizer, ElectraModel

tokenizer = ElectraTokenizer.from_pretrained("damien-ir/kosentelectra-discriminator-v2")
model = ElectraModel.from_pretrained("damien-ir/kosentelectra-discriminator-v2")
```

## How to Finetuning / Benchmark
1. 다음의 명령어를 쉘 창에서 입력하여 이 저장소의 파일을 복사합니다.
```
git clone htts://github.com/damien-ir/KoSentELECTRA
cd KoSentELECTRA
```

2. 학습용 데이터와 검증용 데이터를 clone한 KoSentELECTRA에 넣어줍니다.

기본 설정 상으로 파일의 이름은 NSMC의 파일 이름과 같은 ratings_train.txt, ratings_test.txt이며,

다른 파일 이름을 사용하시려면 config.json 파일을 수정해주세요.  
```
cp ../ratings_* .
```

3. 그 후 docker-comppose 를 실행하여 도커를 컨테이너화 합니다.

docker-compose.yml에서 미리 설정을 해두었기 때문에,
성공적으로 컨테이너화 할 경우 자동으로 모델의 학습이 시작됩니다.
```
docker-compose -f docker-compose.yml up
```

4. 이후 셸 창에서 모델의 학습 과정을 지켜볼 수 있습니다.

학습 과정이 궁금하지 않으시다면, 명령어 맨 뒤에 -d를 붙여 백그라운드로 실행해주세요.
```
docker-compose -f docker-compose.yml up -d
```

5. docker를 이용한 학습이 싫으시다면, 직접 ```classification.py``` 를 실행하여 fine-tuning / benchmark를 실행할 수 있습니다.

## About Model's Corpus
개인으로서 수집할 수 있는 대용량 말뭉치는 다 사용하였으며, 각 모델마다 차이가 있습니다.
사용한 코퍼스 목록들은 다음과 같습니다.
* [KcBERT Corpus](https://www.kaggle.com/junbumlee/kcbert-pretraining-corpus-korean-news-comments)
* [Wikipedia](https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4_%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C)
* [NSMC](https://github.com/e9t/nsmc)
* [KCC](http://nlp.kookmin.ac.kr/kcc/)
* [국어국립원 모두의 말뭉치](https://corpus.korean.go.kr/) 구어 / 문어 / 신문 / 메신저 / 웹 말뭉치
* 직접 크롤링한 음식점 리뷰 (주로 구어체, 21GB)

각 모델 별 Vocab와 말뭉치의 차이는 다음과 같습니다.


모든 vocab는 tokeniers의 BertWordPieceTokenizer를 사용하여 만들었으며,

limit_alphabet을 1만 이상으로 설정하여 생성하였습니다

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

```
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

## Benchmark Result
배치 사이즈, 학습률 등의 설정을 조정하여 NSMC 태스크에서 최고 정확도를 91.49%까지 달성하였습니다.

S3에 업로드 되어 있는 모델의 성능은 91.35% 이므로 해당 모델 사용에 오해 없으시길 바랍니다.

![Benchmark Result](https://raw.githubusercontent.com/Damien-IR/KoSentELECTRA/master/images/benchmark_result.png)


## Acknowledgement
TensorFlow Research Cloud(TFRC) 의 지원을 받아 Cloud TPU로 모델을 학습하였습니다.<br>

## Reference
- [ELECTRA](https://github.com/google-research/electra)
- [KcBERT](https://github.com/Beomi/KcBERT)
- [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)
- [kor_pretrain_LM](https://github.com/enlipleai/kor_pretrain_LM)
