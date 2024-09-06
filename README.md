# ❤️ Subtle Emotion Recognition Using CLIP Model

**Team SOㅋTA**  
*Members: Park Se-hoon, Bae Ji-won, Jeong Ha-yeon*  

## 🚀 Overview

Welcome to the Emotion Recognition Using CLIP Model project. This project, developed by Team SOㅋTA, explores the use of a fine-tuned CLIP model to perform **zero-shot prediction** for complex emotions that were not part of the training labels. Our primary goal was to leverage CLIP's strengths in multi-modal tasks to identify emotions in images that map to text-embedded emotions.

We observed that many emotion datasets for facial expressions have limited and overly simplistic emotion labels. Our motivation was to develop a model that could recognize more nuanced emotions and ultimately create a dataset that could serve broader purposes, such as psychological analysis or improving AI-human interactions.

---

## 💡 Project Objectives

1. Fine-tune the **CLIP model** for zero-shot emotion recognition.
2. Expand emotion labels beyond simple categories using **image feature embeddings** linked to **text embeddings**.
3. Create a dataset with more granular emotional labels to enhance emotion recognition capabilities.

---

## 🔎 Methodology

### 1. Data Collection

We used two main datasets:
- **FER-2013**: A dataset containing 35,887 grayscale images labeled with 7 basic emotions (anger, disgust, fear, happiness, sadness, surprise, and neutral).
- **AffectNet**: A dataset with over 1 million images labeled with 8 emotions (same as FER-2013 plus contempt).

We merged these datasets to ensure the availability of a sufficient number of images for each emotion class. The **Kaggle dataset API** was used to download and merge the datasets while ensuring label balance in each batch.

### 2. Model Architecture

We chose the CLIP model due to its ability to connect **text and image embeddings**. For our task, we modified the model architecture by:
- Adding depth to the image encoder and classifier layers.
- Using **Ray Tune** and manual hyperparameter tuning to optimize parameters such as learning rate, logit scale, and embedding size.
- Fine-tuning both text and image encoders to ensure they aligned with our more complex emotion labels.

### 3. Metrics and Approach

To calculate the **affection score**:
1. **Turn-wise Affection Scoring**: Each turn's affection score is computed using the probability of each emotion label and its associated weight.
2. **Contextual Conversation Scoring**: The overall affection score is computed by summing the product of the probabilities and weights of emotion labels across turns, with greater weight given to recent turns (turn number/total turns).

We used **KoBERT word embeddings** to calculate the cosine similarity between emotions like ‘Excitement,’ ‘Interest,’ ‘Affection,’ and ‘Love,’ and applied these as part of our scoring methodology.

---

## 🧑‍💻 Training Process

1. **Small Dataset Training**: We initially trained a small dataset of 256 images for 100 epochs, achieving a reduction in loss from 2.7 to 1.2 and an improvement in accuracy from 0.16 to 0.40.
2. **Hyperparameter Tuning**: We manually implemented random search to find the optimal combination of parameters, including learning rate, logit scale, and embedding size.
3. **Model Testing**: Using the KOTE dataset, we aimed to classify more complex emotions like "Confused," "Resigned," and "Dignified," but meaningful results have yet to be achieved.

---

## Limitations and Future Work

- We faced challenges in achieving meaningful accuracy with more complex emotions and are currently exploring ways to improve the model by experimenting with different architectures such as replacing **ViT** with **ResNet**.
- Our next task is to focus on classifying 36 new emotions defined in the Korean language corpus and calculating **top-5 accuracy** for these categories.

---

## Conclusion

While we faced several challenges in tuning the CLIP model for more complex emotions, this project laid the groundwork for future research. We aim to continue refining the model and extend its application to diverse fields such as psychological analysis and interactive AI systems.

---

## Acknowledgments

- This project was created by Team SOㅋTA: **Park Se-hoon, Bae Ji-won, Jeong Ha-yeon**.

---

# ❤️ CLIP 모델을 활용한 감정 인식

**Team SOㅋTA**  
*팀원: 박세훈, 배지원, 정하연*

## 🚀 개요

'CLIP 모델을 활용한 감정 인식' 프로젝트 리포지토리에 오신 것을 환영합니다. 이 프로젝트는 Team SOㅋTA가 개발한 것으로, **zero-shot prediction**을 통해 학습되지 않은 복잡한 감정을 인식하는 모델을 목표로 했습니다. 저희는 멀티모달 태스크에 강점이 있는 CLIP 모델을 파인튜닝하여 텍스트 임베딩과 이미지 피쳐 임베딩을 연결하는 모델을 개발하고자 했습니다.

저희는 기존 표정 데이터셋에서 사용되는 감정 레이블이 단순하다는 문제를 발견하고, 더 미묘한 감정을 인식할 수 있는 모델을 개발하고자 했습니다. 이를 통해 세분화된 감정 레이블을 활용한 데이터셋을 만들어 심리 분석 및 AI-인간 상호작용에 기여하고자 했습니다.

---

## 💡 프로젝트 목표

1. **CLIP 모델**을 파인튜닝하여 zero-shot 감정 인식을 가능하게 합니다.
2. 이미지와 텍스트 간 임베딩을 연결하여 더 복잡한 감정 레이블을 구분할 수 있는 모델을 구축합니다.
3. 더 세분화된 감정 데이터셋을 만들어 배포하는 것을 목표로 합니다.

---

## 🔎 방법론

### 1. 데이터 수집

저희는 두 가지 주요 데이터셋을 사용했습니다:
- **FER-2013**: 35,887개의 흑백 이미지로 구성된 데이터셋으로, 7가지 기본 감정(화남, 혐오, 두려움, 행복, 슬픔, 놀람, 중립)으로 라벨링됨.
- **AffectNet**: 1백만 개 이상의 이미지로 구성된 데이터셋으로, 8가지 감정(기본 감정 + 경멸)으로 라벨링됨.

저희는 이 데이터셋을 합쳐서 충분한 이미지 데이터를 확보하고, **Kaggle dataset API**를 이용하여 데이터를 통합하고 배치마다 라벨의 균형을 맞췄습니다.

### 2. 모델 아키텍쳐

저희는 CLIP 모델을 선택하였으며, 텍스트 임베딩과 이미지 피쳐 간의 연결을 시도했습니다. 
- 이미지 인코더와 분류기의 레이어 깊이를 추가.
- **Ray Tune** 및 수동 하이퍼파라미터 튜닝을 사용하여 학습률, 로짓 스케일, 임베딩 크기 등 최적의 파라미터를 찾았습니다.
- 더 복잡한 감정 라벨을 학습할 수 있도록 텍스트와 이미지 인코더 모두를 파인튜닝했습니다.

### 3. 메트릭 및 접근 방식

**호감도 점수**를 계산하는 방법은 다음과 같습니다:
1. **턴별 호감도 점수 산출**: 각 턴의 호감도 점수는 감정 레이블의 확률과 가중치를 곱하여 계산.
2. **맥락을 고려한 대화 분위기 파악**: 대화의 전체 분위기는 감정 레이블 확률과 가중치를 더한 후, 최신 턴에 더 큰 가중치를 부여하여 계산.

저희는 **KoBERT 워드 임베딩**을 사용하여 ‘설렘’, ‘관심’, ‘호감’, ‘사랑’ 등 감정 라벨 간의 코사인 유사도 점수를 계산하였습니다.

---

## 🧑‍💻 학습 과정

1. **소규모 데이터셋 학습**: 256개의 이미지로 구성된 small dataset으로 100 epoch를 학습하여, loss가 2.7에서 1.2로 감소하고, accuracy가 0.16에서 0.40까지 증가하는 것을 확인했습니다.
2. **하이퍼파라미터 튜닝**: 학습률, 로짓 스케일, 임베딩 크기를 최적화하기 위해 random search를 수동으로 구현했습니다.
3. **모델 테스트**: **KOTE 데이터셋**을 사용하여 더 복잡한 감정(예: "당황", "한심", "비장함")을 분류하려 했지만, 아직 유의미한 결과는 도출하지 못했습니다.

---

## 한계점 및 향후 작업

- 더 복잡한 감정을 학습하는 데 어려움을 겪었고, **ViT** 대신 **ResNet** 등 다른 아키텍처를 실험하고 있습니다.
- 한국어 말뭉치 데이터셋에서 정의된 36개의 새로운 감정을 분류하고 **top-5 accuracy**를 계산하는 것이 저희의 다음 목표입니다.

---

## 결론

이번 프로젝트는 복잡한 감정 인식을 위한 CLIP 모델 튜닝의 기초를 마련했습니다. 향후 심리 분석, AI-인간 상호작용 등 다양한 분야에 이 모델을 확장해 나갈 계획입니다.

---

## 감사의 말

- 이 프로젝트는 Team SOㅋTA: **박세훈, 배지원, 정하연**이 제작하였습니다.

