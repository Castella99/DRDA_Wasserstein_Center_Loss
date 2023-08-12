# 2023 한국컴퓨터종합학술대회 [EEG 분류를 위한 와서스테인 거리 손실을 사용한 심층 표현 기반의 도메인 적응 기법]

![Model Overview](W-DRDA_model_overview.jpeg)

파일 구성 : main.sh / main.py / model.py / loadData.py / train_val_model.py

main.sh : Hyperparameter 튜닝을 위한 리눅스 쉘스크립트, 파일 실행을 위해서는 리눅스 기반 환경 구성이 필요.

main.py : 메인 파이썬 코드. 모델 학습을 위해 사용됨

model.py : 모델 파이썬 코드

loadData.py : 데이터 (DEAP)를 불러오고 데이터 전처리를 수행

train_val_model.py : 학습 및 검증, 테스트에 필요한 함수 내장


