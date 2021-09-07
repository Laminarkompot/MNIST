import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
from PyQt5 import QtWidgets, QtGui

# Задние констант
DATA_PATH = '/home/den/code/edu/mnist/'
NET_PATH = '/home/den/code/edu/mnist/'
DEVICE = 'cuda'

# Чтение данных
data = pd.read_pickle(DATA_PATH + 'train.pkl')
X_train = data.iloc[:21000,1:].values

# Расчёт параметров нормализации данных
mean = X_train.mean()
std = X_train.std()

# Загрузка обученной модели
model = torch.load(NET_PATH + 'ConvNet')

# Передача модели в видеопамять
model.to(DEVICE)

# Функция инференса
def launch_inference(i):
    label = data.iloc[i, 0]
    digit = data.iloc[i, 1:].values
    digit = (digit - mean) / std
    imdigit = digit.reshape(28, 28)
    digit = torch.tensor(digit, dtype=torch.float32, device=DEVICE)
    prediction = model(digit.reshape(1, 1, 28, 28))
    prediction = prediction.detach().cpu().numpy()
    prediction = np.argmax(prediction)
    fig = plt.figure()
    plt.imshow(imdigit, cmap='binary')
    plt.title('Правильный ответ: ' + str(label) + '     Ответ нейросети: ' + str(prediction))
    fig.savefig('number.png')

# Функция без параметров, вызывающая функцию инференса и обновляющая картинку
def action():
    lbl.clear()
    launch_inference(int(i.text()))
    pic = QtGui.QPixmap(NET_PATH + 'number.png')
    lbl.setPixmap(pic)

# Задание приложения и окна
app = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QWidget()
window.setWindowTitle('Утилита тестирования свёрточной нейросети')
window.resize(700, 100)

# Задание содержимого окна
label = QtWidgets.QLabel('Введите порядковый номер элемента выборки от 0 до 41999:')
num = 8
i = QtWidgets.QLineEdit(str(num))
launch_inference(int(i.text()))
btnQuit = QtWidgets.QPushButton('Закрыть окно')
btnCompute = QtWidgets.QPushButton('Рассчитать')
lbl = QtWidgets.QLabel()
pic = QtGui.QPixmap(NET_PATH + 'number.png')
lbl.setPixmap(pic)

# Задание расположения содержимого в окне
vbox = QtWidgets.QVBoxLayout()
vbox.addWidget(label)
vbox.addWidget(i)
vbox.addWidget(btnCompute)
vbox.addWidget(lbl)
vbox.addWidget(btnQuit)
window.setLayout(vbox)

# Задание реакций на нажатия кнопок
btnCompute.clicked.connect(action)
btnQuit.clicked.connect(app.quit)

window.show()

# Запуск основного цикла приложения
sys.exit(app.exec_())

