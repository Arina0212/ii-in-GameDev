# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Шубина Арина Николаевна
- РИ-210947

Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
### Написать программы Hello World на Python и Unity.
- Python
```py
 Print("Hello World")
 ```
![2022-09-23_23-57-40](https://user-images.githubusercontent.com/114181560/192038303-c05ce954-1458-4223-85d0-47e8fa40b32c.png)
-Unity
```py
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HelloWorld : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("Hello World");
    }
}
```
![2022-09-24_00-21-49](https://user-images.githubusercontent.com/114181560/192042204-c693c13b-4d34-4d91-b9f5-41dd5523ae4e.png)

## Задание 2
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задач
Ход работы:
- Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py

In [ ]:
#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt
#This code is for jupyter Notebook only
%matplotlib inline

# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)
```
![2022-09-24_00-41-07](https://user-images.githubusercontent.com/114181560/192045035-6bc73dd2-da72-4027-875e-2875a97e887d.png)

```py
#Show the effect of a scatter plot
plt.scatter(x,y)

#The basic linear regression model is wx+ b, and since this is a two-dimensional space, the model is ax+ b

def model(a, b, x):
    return a*x + b

#Tahe most commonly used loss function of linear regression model is the loss function of mean variance difference
def loss_function(a, b, x, y):
    num = len (x)
    prediction=model (a,b, x)
    return (0.5/num) * (np.square(prediction-y)).sum()

#The optimization function mainly USES partial derivatives to update two parameters a and b
def optimize (a,b,x,y):
    num = len (x)
    prediction = model (a,b, x)
    
    #Update the values of A and B by finding the partial derivatives of the loss function on a and b
    da = (1.0/num) * ((prediction -y)*x).sum()
    db = (1.0/num) * ((prediction -y).sum())
    a =a - Lr*da
    b= b - Lr*db
    return a, b

#iterated function, return a and b
def iterate(a,b,x,y, times) :
    for i in range (times):
        a,b = optimize(a,b,x,y)
    return a,b
#Initialize parameters and display
a = np.random.rand(1)
print (a)
b = np.random.rand(1)
print (b)
Lr = 0.000001
#For the first iteration, the parameter values, losses, and visualization after the iteration are displayed
a,b = iterate(a,b,x,y,1)
prediction=model (a,b, x)
loss = loss_function(a, b, x, y)
print (a,b, loss)
plt.scatter (x,y)
plt.plot (x,prediction)
```
![2022-09-24_00-49-44](https://user-images.githubusercontent.com/114181560/192046346-388d6430-3695-4d51-898c-37ac848c8c15.png)

```py

a,b = iterate(a,b,x,y,2)
prediction=model (a,b, x)
loss = loss_function(a, b, x, y)
print (a,b, loss)
plt.scatter (x,y)
plt.plot (x,prediction)

```

![2022-09-24_00-42-43](https://user-images.githubusercontent.com/114181560/192045306-6abb31b1-4dcc-41bf-b8f4-a494054c2840.png)

```py

a,b = iterate(a,b,x,y,3)
prediction=model (a,b, x)
loss = loss_function(a, b, x, y)
print (a,b, loss)
plt.scatter (x,y)
plt.plot (x,prediction)
```
![image](https://user-images.githubusercontent.com/114181560/192046448-718bd810-33ed-4d9b-87ef-60faaa49f810.png)

```py

a,b = iterate(a,b,x,y,4)
prediction=model (a,b, x)
loss = loss_function(a, b, x, y)
print (a,b, loss)
plt.scatter (x,y)
plt.plot (x,prediction)
```
![image](https://user-images.githubusercontent.com/114181560/192046624-35cdad50-a5a0-4b35-bddf-80be9758437b.png)

```py

a,b = iterate(a,b,x,y,5)
prediction=model (a,b, x)
loss = loss_function(a, b, x, y)
print (a,b, loss)
plt.scatter (x,y)
plt.plot (x,prediction)
```
![image](https://user-images.githubusercontent.com/114181560/192046679-9d49d84e-7cf1-4ca9-b62b-83e9dc183cbc.png)

```py

a,b = iterate(a,b,x,y,1000)
prediction=model (a,b, x)
loss = loss_function(a, b, x, y)
print (a,b, loss)
plt.scatter (x,y)
plt.plot (x,prediction)
```
![2022-09-24_00-44-19](https://user-images.githubusercontent.com/114181560/192045516-13c2d6c3-a369-4976-b9e5-a767648876a5.png)
