os.getcwd()
os.chdir('C:\\Users\\kitcoop\\.spyder-py3\\분석')
# =============================================================================
#  딥러닝
# =============================================================================
run deep1

# [로지스틱 회귀]
 - 이진 클래스의 분류

activaton() - 입력 신호를 변환하는 장치
 * sigmoid - 0과 1 신호로 변환
 0과 1 사이 구간에 대해 결정 포인트 있어야한다.

로지스틱 회귀의 계수 : 임의값을 경사하강법을 통해 오차를 최소로하는 방식

# 예제) 로지스틱 회귀 정의
# 공부 시간에 따른 합격 여부 

import tensorflow as tf 
import numpy as np 
 
# x, y의 데이터 값 
data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]] x_data = [x_row[0] for x_row in data] y_data = [y_row[1] for y_row in data] 
 
# a와 b의 값을 임의로 정한다.
a = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0)) 
 
# y 시그모이드 함수의 방정식을 세운다.
y = 1/(1 + np.e**(a * x_data + b))

# loss를 구하는 함수 (오차를 줄이는 방향으로 기울기 수정해야한다.)
loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1 - y)) 
 
# 학습률 값
learning_rate = 0.5 
 
# loss를 최소로 하는 값 찾기
 gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) 
 
# 학습
 with tf.Session() as sess: 
     sess.run(tf.global_variables_initializer()) 
     
 
 for i in range(60001):   
      sess.run(gradient_decent)  
       if i % 6000 == 0:         
           print("Epoch: %.f, loss = %.4f, 기울기 a = %.4f, y 절편 = %.4f" %                      (i, sess.run(loss), sess.run(a), sess.run(b)) 

# tf.placeholder : 값을 저장하기 위한 그릇을 만드는 함수
# tf.matmul : 매트릭스 곱하는 함수

# 신경망 활성화 함수
sigmoid : 0 또는 1로 전달 (out put layer : 이진 class 분류)
softmax : out put layer 3개 이상 class 분류
relu    : 0 또는 x(입력값)값 그대로 전달(중간층 layer)

 # softmax
 - 3개 이상 class 분류.
 - 노드를 입력 받고 제일 큰 신호는 1, 그 외 신호는 0으로 분류 (신호의 크기에 중점)
 - Y의 형태에 따라 out put layer 갯수 달라진다.
  
 
# =============================================================================
#  # ANN - iris data
# =============================================================================
run profile1
import tensorflow as tf
import keras

from keras.models import Sequential 
from keras.layers.core import Dense 
from keras.utils import np_utils 
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('iris.csv') 
df.columns

# 데이터 분류 
dataset = df.values 
X = dataset[:,0:4].astype(float) 
Y_obj = dataset[:,4] 
 
# 문자열을 숫자로 변환 
e = LabelEncoder() 
e.fit(Y_obj) 
Y = e.transform(Y_obj) 
Y_encoded = np_utils.to_categorical(Y) 

# 모델의 설정 
model = Sequential() 
model.add(Dense(16, input_dim=4, activation='relu')) 
model.add(Dense(3, activation='softmax')) 
 
# 모델 컴파일  
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) 
 
# 모델 실행 
model.fit(X, Y_encoded, epochs=50, batch_size=1) 
 
# 결과 출력  
print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))

# =============================================================================
# ANN - cancer data
# =============================================================================
run profile1
os.getcwd()
os.chdir('C:\\Users\\kitcoop\\.spyder-py3\\분석')


run profile1
import tensorflow as tf
import keras 

from keras.models import Sequential 
from keras.layers.core import Dense 
from keras.utils import np_utils 
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('cancer.csv')
df.columns.shape

# 분류
dataset = df.values
x = dataset[:,2:32]
y_obj = dataset[:,1]

# 문자열 숫자로 변환
e = LabelEncoder()
e.fit(y_obj)
Y = e.transform(y_obj)
Y_encoded = np_utils.to_categorical(Y)
len(df.columns)
# 모델 설정
model = Sequential()
model.add(Dense(32, input_dim=30, activation = 'relu'))
model.add(Dense(15))
model.add(Dense(8))
model.add(Dense(2, activation='sigmoid'))

# 모델 컴파일
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

# 모델 실행
model.fit(x,Y_encoded, batch_size=1, epochs=100)

# 결과출력
print("\n Accuracy : %.4f" % (model.evaluate(x,Y_encoded)[1]))
