# İŞ PROBLEMİ

# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.

# VERİ SETİ HİKAYESİ

# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu veİnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.



# GÖREV 1: KEŞİFÇİ VERİ ANALİZİ


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.

df=pd.read_csv("8.Hafta/Telco-Customer-Churn.csv")
df.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum()) #eksik deger var mı? varsa kac tane?
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T) # sayısal değişkenlerin ceyrekliklerinin incelenmesi

check_df(df)




# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

df.info()
df.head()

df["Partner"]=df["Partner"].apply(lambda x:1 if x == "Yes" else 0)
df["Dependents"]=df["Dependents"].apply(lambda x:1 if x == "Yes" else 0)
df["PhoneService"]=df["PhoneService"].apply(lambda x:1 if x == "Yes" else 0)
df["PaperlessBilling"]=df["PaperlessBilling"].apply(lambda x:1 if x == "Yes" else 0)
df["Churn"]=df["Churn"].apply(lambda x:1 if x == "Yes" else 0)



# Adım 3:  Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.


def cat_summary(dataframe, col_name, plot=False): # plot:true olursa if çalışır.
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),  #değişkende hangi degerden kacar adet var?
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})) # deger adetlerini toplam deger sayısına bölümü oran verir.
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


cat_summary(df, "Churn")

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Churn", col)

# Adım 5: Aykırı gözlem var mı inceleyiniz.


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit



def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(check_outlier(df, col))




# Adım 6: Eksik gözlem var mı inceleyiniz.

df.isnull().sum()

# Görev 2 : Feature Engineering



# Adım 1:  Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

# aykırı veya eksik gözlem yok (varmış aşağıda düzelttim.)

# Adım 2: Yeni değişkenler oluşturunuz.





# Adım 3:  Encoding işlemlerini gerçekleştiriniz.


#label encoding

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

df.info()

# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi
# target değişkenimi cıkarıyorum.
# bir de binary_cols, zaten daha öncesinde label encoder uygulamıstım.
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)


df.head()

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

num_cols

scaler = RobustScaler() # RobutScaler kullandım çünkü aykırı değerlerden daha az etkileniyor.
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head(20)
df.shape

df.describe()

#Görev 3 : Modelleme



# Adım 1:  Sınıflandırma algoritmaları ile modeller kurup, accuracyskorlarını inceleyip. En iyi 4 modeli seçiniz.


# KNN

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

df.info()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.loc[df["TotalCharges"].isnull(), "TotalCharges"] = df[col].median()


knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)

# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
# acc 0.83
# f1 0.62
# AUC
roc_auc_score(y, y_prob)
# 0.87

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.7476941052648558
cv_results['test_f1'].mean()
#0.43
cv_results['test_roc_auc'].mean()
#0.68



# Adım 2: Seçtiğiniz modeller ile hiperparametreoptimizasyonu gerçekleştirin ve bulduğunuz hiparparametrelerile modeli tekrar kurunuz.

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50),
              "leaf_size": (20,30,40,50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_gs_best.best_params_



################################################
# 6. Final Model
################################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

random_user = X.sample(1)

knn_final.predict(random_user)



