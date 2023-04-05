from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
 
dataset_ivt_11_22 = pd.read_csv("main/database/students_ivt_11_22.csv")
dataset_ivt_22_22 = pd.read_csv("main/database/students_ivt_22_22.csv")
dataset_ivt_41_22 = pd.read_csv("main/database/students_ivt_41_22.csv")
dataset_ivt_42_22 = pd.read_csv("main/database/students_ivt_42_22.csv")
dataset_ivt_43_22 = pd.read_csv("main/database/students_ivt_43_22.csv")
dataset_kt_31_22 = pd.read_csv("main/database/students_kt_31_22.csv")
dataset_kt_41_22 = pd.read_csv("main/database/students_kt_41_22.csv")
dataset_kt_42_22 = pd.read_csv("main/database/students_kt_42_22.csv")
 
models_dict = {}
all_groups = ["ivt_11_22", "ivt_22_22", "ivt_41_22", "ivt_42_22", "ivt_43_22", "kt_31_22", "kt_41_22", "kt_42_22"]
all_files = [dataset_ivt_11_22, dataset_ivt_22_22, dataset_ivt_41_22, dataset_ivt_42_22, dataset_ivt_43_22,
             dataset_kt_31_22, dataset_kt_41_22, dataset_kt_42_22]
russian_names = ["ИВТ-11-22","ИВТ-22-22","ИВТ-41-22","ИВТ-42-22","ИВТ-43-22","КТ-31-22","КТ-41-22", "КТ-42-22"]
 
for i in range(len(all_files)):
    group = all_groups[i]
    group_dataset = all_files[i]
    X = group_dataset.drop(columns=["оценка"], axis=1)  # извлекаем предикторы
    y = group_dataset["оценка"]  # извлекаем ответ
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential([
        Dense(32, input_shape=(7,), activation='sigmoid'),
        Dense(16, input_dim=4, activation="relu"),
        Dense(8, activation="relu"),
        Dense(4, activation="sigmoid"),
        Dense(1, activation="linear")
    ])
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error", "accuracy"])
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1, validation_split=0.1)
    models_dict[group] = model
 
 
def ege_counter(object, group):
    dataset = all_files[all_groups.index(group)]
    ax = sns.histplot(dataset[object], binwidth=0.99)
    ax.set(xlabel='балл ЕГЭ по предмету {}'.format(object),
           ylabel='количество сдавших',
           title=russian_names[all_groups.index(group)])
    plt.show()
 
 
def get_predict(group, math, inf, rus, region, bvi, chgu, hope):
    model = models_dict[group]
    df = pd.DataFrame(
        {"русский язык": [rus], "математика": [math], "информатика": [inf], "регион": [region], "перечень": [bvi],
         "олимпиада чгу": [chgu], "надежда": [hope]})
    prediction = model.predict(df)
    return prediction



summ_mean_ege_dict = {}
 
region_counter_dict = {}
 
bvi_counter_dict = {}
 
chgu_counter_dict = {}
 
hope_counter_dict = {}
 
mean_ball_olymp_dict = {}
 
def summ_mean_ege(group):
    if group not in summ_mean_ege_dict:
        dataset = all_files[all_groups.index(group)]
        summ_mean_ege_dict[group] = round((dataset["русский язык"] + dataset["математика"] + dataset["информатика"]).mean(),2)
    return summ_mean_ege_dict[group]
 
 
def region_counter(group):
    if group not in region_counter_dict:
        dataset = all_files[all_groups.index(group)]
        region_counter_dict[group] = dataset['регион'].sum() 
    return region_counter_dict[group]
 
 
def bvi_counter(group):
    if group not in bvi_counter_dict:
        dataset = all_files[all_groups.index(group)]
        bvi_counter_dict[group] = dataset['перечень'].sum()
    return bvi_counter_dict[group]
 
 
 
def chgu_counter(group):
    if group not in chgu_counter_dict:
        dataset = all_files[all_groups.index(group)]
        chgu_counter_dict[group] = dataset['олимпиада чгу'].sum()
    return chgu_counter_dict[group]
 
 
 
def hope_counter(group):
    if group not in hope_counter_dict:
        dataset = all_files[all_groups.index(group)]
        hope_counter_dict[group] = dataset['надежда'].sum()
    return hope_counter_dict[group] 
 
 
def mean_ball_olymp(group):
    dataset = all_files[all_groups.index(group)]
    if group not in mean_ball_olymp_dict:
        sm = 0
        cnt = 0
        for i in range(0, 300000):
            if (dataset["перечень"].iloc[i] + dataset["олимпиада чгу"].iloc[i] + dataset["надежда"].iloc[i] +
                    dataset["регион"].iloc[i] > 0):
                sm += (dataset["русский язык"].iloc[i] + dataset["математика"].iloc[i] + dataset["информатика"].iloc[i])
                cnt += 1
        mean_ball_olymp_dict[group] = round(sm / cnt, 2)
    return mean_ball_olymp_dict[group]
 
def precalc():
    for i in all_groups:
        summ_mean_ege(i)
        region_counter(i)
        bvi_counter(i)
        chgu_counter(i)
        hope_counter(i)
        mean_ball_olymp(i)

precalc()


def index(request):
    data = {
        'title': 'Главная страница'
    }
    return render(request, 'main/index.html', data)

def answer(request):
    math = int(request.POST.get("math", "Undefined"))
    rus = int(request.POST.get("rus", "Undefined"))
    inf = int(request.POST.get("inf", "Undefined"))
    group = request.POST.get("groupRadio", "Undefined")
    vsosh = int(request.POST.get("vsosh", "0"))
    perech = int(request.POST.get("perech", "0"))
    chuvsu_olimp = int(request.POST.get("chuvsu-olimp", "0"))
    nadezh = int(request.POST.get("nadezh", "0"))

    predict = get_predict(group, math, inf, rus, vsosh, perech, chuvsu_olimp, nadezh)
    olympArr = []
    if (vsosh == 1):
        olympArr.append("Региональный этап всроссийской олимпиады школьников")
    if (perech == 1):
        olympArr.append("Перечневые олимпиады")
    if (chuvsu_olimp == 1):
        olympArr.append("Олимпиада чувашского государственного университета")
    if (nadezh == 1):
        olympArr.append("Олимпиада «Надежда машиностроения Чувашии» / «Надежда электротехники Чувашии»")


    data = {
        'title': 'Учебная статистика',
        'math': math,
        'rus': rus,
        'inf': inf,
        'group': russian_names[all_groups.index(group)],
        'groupEng': group,
        'olymp': olympArr,
        'summ_mean_ege': summ_mean_ege(group),
        'region_counter': region_counter(group),
        'bvi_counter': bvi_counter(group),
        'chgu_counter': chgu_counter(group),
        'hope_counter': hope_counter(group),
        'mean_ball_olymp': mean_ball_olymp(group),
        'predict': predict[0][0]
    }
    
    return render(request, 'main/answer.html', data)
