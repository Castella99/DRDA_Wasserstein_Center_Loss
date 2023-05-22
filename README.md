<h2>2023 KCC 제출 EEG 분류를 위한 와서스테인 거리 손실을 사용한 심층 표현 기반의 도메인 적응 기법 코드</h2>

파일 구성 : main.sh / main.py / model.py / loadData.py / train_val_model.py

main.sh : Hyperparameter 튜닝을 위한 리눅스 쉘스크립트, 파일 실행을 위해서는 리눅스 기반 환경 구성이 필요.

main.py : 메인 파이썬 코드. 모델 학습을 위해 사용됨

model.py : 모델 파이썬 코드

loadData.py : 데이터 (DEAP)를 불러오고 데이터 전처리를 수행

train_val_model.py : 학습 및 검증, 테스트에 필요한 함수 내장

코드 실행 방법

1. 리눅스 환경일 경우

Hyperparameter를 튜닝하기 위해 for문을 통해 튜닝하고 싶은 값을 넣어 각각의 성능을 비교
Hyperparameter mu값과 lambada 값 튜닝

sh main.sh로 실행

연구실 GPU 서버 이용 시 사용 바람

2. 이외 환경 (파이썬 코드만 단독 실행)

python -u main.py -e={Epochs} -p={Early Stopping Patience} -b={Batch Size} -l={Hyperparameter lambda} -m={Hyperparameter mu} -n={Hyperparameter n} -k={fold} -t={test size}

*** main.py 내에 있는 path 변수를 반드시 로컬 DEAP 저장위치로 변경하길 바람 ***
Repository Directory를 DEAP dataset(data_preprocessed_matlab) Directory와 같은 Directory에 위치시키면 변경할 필요 없음 (형제 디렉토리로 배치할 것)

아나콘다 환경설정 가이드

Name                      Version                   Build  Channel <br>
_libgcc_mutex             0.1                        main  <br>
_openmp_mutex             5.1                       1_gnu  <br>
ca-certificates           2023.01.10           h06a4308_0  <br>
contourpy                 1.0.7                    pypi_0    pypi <br>
cycler                    0.11.0                   pypi_0    pypi <br>
fonttools                 4.39.3                   pypi_0    pypi <br>
importlib-resources       5.12.0                   pypi_0    pypi <br>
joblib                    1.2.0                    pypi_0    pypi <br>
kiwisolver                1.4.4                    pypi_0    pypi <br>
ld_impl_linux-64          2.38                 h1181459_1  <br>
libffi                    3.3                  he6710b0_2  <br>
libgcc-ng                 11.2.0               h1234567_1  <br>
libgomp                   11.2.0               h1234567_1  <br>
libstdcxx-ng              11.2.0               h1234567_1  <br>
matplotlib                3.7.1                    pypi_0    pypi <br>
ncurses                   6.4                  h6a678d5_0  <br>
numpy                     1.24.2                   pypi_0    pypi <br>
openssl                   1.1.1t               h7f8727e_0  <br>
packaging                 23.1                     pypi_0    pypi <br>
pandas                    2.0.0                    pypi_0    pypi <br>
pillow                    9.5.0                    pypi_0    pypi <br>
pip                       23.0.1           py38h06a4308_0  <br>
pyparsing                 3.0.9                    pypi_0    pypi <br>
python                    3.8.10               h12debd9_8  <br>
python-dateutil           2.8.2                    pypi_0    pypi <br>
pytz                      2023.3                   pypi_0    pypi <br>
readline                  8.2                  h5eee18b_0  <br>
scikit-learn              1.2.2                    pypi_0    pypi <br>
scipy                     1.10.1                   pypi_0    pypi <br>
setuptools                66.0.0           py38h06a4308_0  <br>
six                       1.16.0                   pypi_0    pypi <br>
sqlite                    3.41.2               h5eee18b_0  <br>
threadpoolctl             3.1.0                    pypi_0    pypi <br>
tk                        8.6.12               h1ccaba5_0  <br>
torch                     1.7.1+cu110              pypi_0    pypi <br>
torchvision               0.8.2+cu110              pypi_0    pypi <br>
tqdm                      4.65.0                   pypi_0    pypi <br>
typing-extensions         4.5.0                    pypi_0    pypi <br>
tzdata                    2023.3                   pypi_0    pypi <br>
wheel                     0.38.4           py38h06a4308_0  <br>
xz                        5.2.10               h5eee18b_1  <br>
zipp                      3.15.0                   pypi_0    pypi <br>
zlib                      1.2.13               h5eee18b_0  <br>