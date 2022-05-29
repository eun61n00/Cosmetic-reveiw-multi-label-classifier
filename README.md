# ***Cosmetic Review Classifier***

## **프로젝트 배경**
리뷰는 현재 e커머스 시장에서 매우 중요한 요소입니다.
특히 화장품 제품군의 경우 화장품의 카테고리(기초, 색조, 바디, 헤어 등)도 다양하고 하나의 카테고리 내에서도 화장품의 품질로 고려되는 요소들(보습력, 지속력, 용기, 디자인, 색상, 성분 등)이 매우 많기 때문에 화장품 리뷰에서 평가된 요소들을 판단해주는 과정이 의미있을 것으로 생각하여 Cosmetic Review Classifier 프로젝트를 진행하게 되었습니다.

<br></br>

## **데이터 수집**
`naver_beautywindow_review_scraping.py`

[네이버 쇼핑 뷰티윈도우](https://shopping.naver.com/beauty/home)의 리뷰들을 크롤링하여 모델을 위한 데이터셋으로 사용하였습니다. </br>
multi label classificaiton 모델을 만들기 위하여 네이버에 이미 분류되어있는 리뷰들을 분류값과 함께 수집하였습니다.
<img width="1391" alt="ref_naver_beautywindow_review" src="https://user-images.githubusercontent.com/71613548/170852084-3b49ec7e-40cd-408f-bef0-9f53a2822395.png">
<br></br>

## **데이터 전처리**
`preprocess.py`
- 수집한 데이터(dictionary)를 dataframe 형태로 변경
- multi-lable 모델을 위해 해당되는 분류값만 1인 배열로 label 컬럼을 생성
- 카테고리에 해당되는 리뷰의 개수가 10000개 이하인 카테고리는 삭제
- 별점 4, 5점은 긍정 리뷰로, 1, 2, 3점은 부정 리뷰로 분류 (전체적으로 긍정적인 리뷰가 많은 것으로 고려)


<br></br>

## **사용 모델 및 Training**
`cosmetic_review_classifier.py`
- SKT에서 공개한 한국어 embedding model인 [KoBERT](https://github.com/SKTBrain/KoBERT) 모델을 사용하였습니다.
- 모델 사용은 https://github.com/myeonghak/kobert-multi-label-VOC-classifier 를 참고하였습니다.
- optimizer는 `AdamW`을 사용하였습니다. ([참고한 githug repository](https://github.com/myeonghak/kobert-multi-label-VOC-classifier)-myeonghak)
- 손실함수는 multi-label classification을 위해 torch에서 제공하는 `BCEWithLogitsLoss()` 함수를 사용하였습니다.
<br></br>

## **Test**
모델을 통해 태깅된 카테고리와 실제 태깅된 카테고리를 비교하면 다음 예시와 같습니다.
<img width="1323" alt="test1" src="https://user-images.githubusercontent.com/71613548/170852091-7e99447e-eac1-47c8-959c-64b97dc699d7.png">
<img width="1329" alt="test2" src="https://user-images.githubusercontent.com/71613548/170852105-892585e8-6717-486b-bbbe-b33066db2739.png">

사용자가 데이터셋에 존재하지않는 리뷰를 직접 모델의 input으로 넣었을 때의 분석 결과는 다음 예시와 같습니다.</br>
<img src='https://user-images.githubusercontent.com/71613548/170851956-ac03dbf4-0b13-4d08-8070-c53b956f5932.png' width=300></br>
<img src = 'https://user-images.githubusercontent.com/71613548/170851958-073d27f8-bc9a-43fb-88bc-d4e2fb5d1a0b.png' width=300></br>
<img src = 'https://user-images.githubusercontent.com/71613548/170851960-6efa61a4-2b85-462c-b99f-7da5458521f7.png' width=300>
</br>

정확도는 다음과 같습니다.
모델의 정확도는 다음과 같습니다.
- Accuracy: 0.7146666666666667
- Column-wise Accuracy: 0.9597199999999999
- micro/precision: 0.8964821416560167
- micro/recall: 0.899932478055368
- micro/f1: 0.8982039963608182
- macro/precision: 0.8443210570023014
- macro/recall: 0.8224970470146014
- macro/f1: 0.8306955336436619
- samples/precision: 0.9024111111111112
- samples/recall: 0.9069444444444446
- samples/f1: 0.8924396825396825}

<br></br>