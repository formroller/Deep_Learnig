

# 경로 확인
import os
os.getcwd()
# 경로변
os.chdir('C:\\Users\\kitcoop\\.spyder-py3\\분석')

import tensorflow as tf
import keras

# =============================================================================
# 딥러닝 - [뉴럴네트워크]
# =============================================================================

from keras.models import Sequential  # Sequential, 순차적, 뼈대 역할 (겹겹이 쌓인 층 만든다)
from keras.layers import Dense       # 뉴럴을 대체하는 노드생성 (레이어)

import numpy 
import tensorflow as tf

# 회귀분석과 유사 (선형회귀가 기본 사상이다.)
# pca 계수는 분산 (Y가 없다.)
# 뉴럴 네트워크 계수는 오차(실제값 - 예측값)에 따라 달라진다

# activation / loss / optimizer -> 종류별 선택


seed = 0 # 고정 시킬때 사용

np.random.seed(seed)

tf.set_random_seed(seed) # X
tf.random.set_seed(0)  # 텐서 시드값 고정 = 계수 초기값 할당하겠다.

Data_set = numpy.loadtxt("ThoraricSurgery.csv", delimiter=",")  # array구조로 바꿔야 하기때문데 loadtxt로 바꾼다.
                                                                # df는 values로 array변경 가능  
Data_set.shape
X = Data_set[:,0:17]
Y = Data_set[:,17]

# 모델 생성 과정 
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))   # 최초 17개 레이어를 30개의 노드로 확장하겠다. (model.add : 1줄이 2개층을 표시한다.)
                                                        # input_dim, x의 갯수 (고정,설명변수 갯수와 일치해야한다.) <> 종속변수 output_dim
                                                        # activation(활성화) : 자극 받을 경우 축적-> 일정 자극 이상 받을 경우 전달
                                                        #                    =>  자극 전달할지 정하는 장치
                                                        #                    =>  노드 뒤에 숨겨진 장치


model.add(Dense(1, activation='sigmoid'))   # Dense 1 사망 여부 결정시 최종 신호는 1개를 받는다. (0 or 1)
                                            # 마지막 레이어의 Dense(노드)수는 학습 갯수에 맞춰야한다.
                                            # Y가 더미변수로 변경될 경우 Y_0,Y_1 이므로 Dense=2


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) # compile 
                                                                                 # loss ='오차함수' *계수 수정시 필요하다 (앞 오차를 어떻게 결정할것인지) (MSE)
                                                                                 # optimaizer *성능 결정 옵션, 손실 함수 값 줄이며 학습하기 위해 사용
                                                                                 # metrics, 모델 성능 평가 기준


model.fit(X, Y, epochs=30, batch_size=10)   # epochs 반복(훈련)횟수 / batch_size 묶음 갯수 10개씩 묶어 30번 반복해라.
                                            # epochs  많을수록 좋다



print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))  # evaluate : sklearn의 score와 같은 함수


# [ MSE ]
# - 뉴럴네트워크 계수 추정하는 기준이 된다.
# - mse 줄이는 방향으로 alpha, beta 도출

# a. 기울기 / b.절편
ab = [3, 76]   # 임의의 계수

# x, y의 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# y = ax + b에 a와 b 값을 대입하여 결과를 출력하는 함수 
def predict(x):
     return ab[0]*x + ab[1]
# => y = 3x +76 

predict(1)

RMSE = sqrt(MSE)

# RMSE 함수 
def rmse(p, a):  
   return np.sqrt(((p - a) ** 2).mean())  # 계산수식
 
# RMSE 함수를 각 y 값에 대입하여 최종 값을 구하는 함수
def rmse_val(predict_result,y):    
    return rmse(np.array(predict_result), np.array(y))   # 실제 값이 들어오는 함수
 
# 예측 값이 들어갈 빈 리스트
 predict_result = [] 
 
# 모든 x 값을 한 번씩 대입하여 
for i in range(len(x)):     
    # predict_result 리스트를 완성한다.     
    predict_result.append(predict(x[i]))    
    print("공부한 시간 = %.f, 실제 점수 = %.f, 예측 점수 = %.f" % (x[i], y[i],     predict(x[i])))
  
## 경사하강법 - 뉴럴네트워크의 계수를 고정할때 사용


# 경사하강법 실습
import tensorflow as tf 
 
# x, y의 데이터 값
 data = [[2, 81], [4, 93], [6, 91], [8, 97]]
 x_data = [x_row[0] for x_row in data]
 y_data = [y_row[1] for y_row in data] 
 
# 기울기 a와 y 절편 b의 값을 임의로 정한다.
 # 단, 기울기의 범위는 0 ~ 10 사이이며, y 절편은 0 ~ 100 사이에서 변하게 한다. 
a = tf.Variable(tf.random_uniform([1], 0, 10, dtype = tf.float64, seed = 0))  # random_uniform, 랜덤하게 데이터 하나 지정 -> 변수화 시켜 담는다(tf.Variable)
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype = tf.float64, seed = 0)) 
 
# y에 대한 일차 방정식 ax+b의 식을 세운다. 
y = a * x_data + b

# 텐서플로 RMSE 함수 
rmse = tf.sqrt(tf.reduce_mean(tf.square( y - y_data )))  # reduce_mean, 평균
 
# 학습률 값 
learning_rate = 0.1  # 어느정도로 수렴하는 구간 찾아갈 것인지(속도 정하는 포인트)
 
# RMSE 값을 최소로 하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse) 
# train 경사하강법 훈련
# rmse를 loss함수로 생각해 값을 최소화


# 텐서플로를 이용한 학습
with tf.Session() as sess:   # with tf.Session()  세션을 연다.
    # 변수 초기화    
    sess.run(tf.global_variables_initializer())   # tf 실행방법 run
    # 2001번 실행(0번째를 포함하므로)  
    for step in range(2001):       
        sess.run(gradient_decent)    
        # 100번마다 결과 출력     
        if step % 100 == 0:       
            print("Epoch: %.f, RMSE = %.04f, 기울기 a = %.4f, y 절편 b = %.4f" % 
                 (step,sess.run(rmse),sess.run(a),sess.run(b))) 
            
            
# =============================================================================
#  다중회귀
# =============================================================================
 - 기울기 수정으로 MSE 줄일 수 없을 경우 새로운 변수의 추가 필요.
 회귀 : 계수 추정하는 방식에 있어 회귀와 비슷하다(MSE)
# =============================================================================
# 로지스틱 회귀 : 분류분석
# =============================================================================
 로지스틱 회귀 : 각 노드가 다음 레이어로 전달될지 여부를 설명
 사용하는 함수 : 시그모이드 함수(0과 1 로 분류, s자 형태로 그래프 그려진다.)
 
 -> 분류기준을 선으로 표시.
 -> 선형 결합으로 0 or 1 분류
 => activation function (f(x)) => Y (0 or 1 신호)
 