####################
## Import modules
####################


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

bakery= pd.read_csv("4-Recommendation Systems/association rules/case3-bakery/Bakery.csv")

bakery.head()
bakery.shape

df = bakery.copy()

###############################
## Get an Idea about dataframe
###############################


def general_info_data(df):
    print("#"*60)
    print("SHAPE")
    print(df.shape)
    print("*" * 60)
    print("TYPE OF COLUMNS & NON-NULL COUNT")
    print(df.info())
    print("#" * 60)
    print("NA ANALYSIS")
    print(df.isna().sum())
    print("*" * 60)
    print("FIRST 10 ROW")
    print(df.head(10))
    print("#" * 60)
    print("COLUMNS NAME")
    print(df.columns)

general_info_data(df)


df_groupby=df.groupby(by=["TransactionNo","Items"])["Items"].count().unstack(). \
            applymap(lambda x: 1 if x > 0 else 0)


####################
## Creating Script
####################

#   To sort the list, this func will be used
def chooseone(liste):
    return liste[0]
#   To draw barplot, this func will be used

def plot_most (x_axis,y_axis,top=10):
    fig,ax = plt.subplots(figsize=(15,30))
    sns.set_theme(style="whitegrid")
    sns_df = pd.DataFrame({"Total Amount": x_axis[:top], "Items": y_axis[:top]})
    sns.set_color_codes("pastel")
    sns.barplot(x="Total Amount", y="Items", data=sns_df)
    for i in ax.containers:
        ax.bar_label(i, )
    plt.show(block=True)

#   To draw bar graph of top 10 items, this func will be used

def count_items(dataframe):
    x_axis=[]
    for i in dataframe.sum():
        x_axis.append(i)
    y_axis = dataframe.columns

    new = list(zip(x_axis,y_axis))

    new.sort(key=chooseone,reverse=True)

    y_a=[]
    x_a=[]
    for x_value,y_value in new:
        x_a.append(x_value)
        y_a.append(y_value)

    plot_most(x_a,y_a)

    return x_a[:20],y_a[:20]

#   By checing total transactions, top 10 the most selling bakery item's bar garph created.
count_items(df_groupby)

#   Data will be seperated dataframe by considering day parts (morning, afternoon, evening & night).
#   This info is taken from original dataframe
#   By using daytype func, data will be seperated or grouped.
def daytype (dataframe,day_type):
    df = dataframe[dataframe["Daypart"] == day_type]
    day_t = df.groupby(["TransactionNo","Items"])["Items"].count().unstack().fillna(0)
    return day_t
morning= daytype(df,"Morning")
afternoon = daytype(df,"Afternoon")
evening = daytype(df,"Evening")
night = daytype(df,"Night")

#   The most selling 20 bakery's item will be listed and top 10 items's bar graph will be created for each day tpe.

x_a_m, y_a_m = count_items(morning)
x_a_a, y_a_a =count_items(afternoon)
x_a_e, y_a_e =count_items(evening)
x_a_n, y_a_n =count_items(night)

#   To compare top 20 selling items by considering day types.

fig,ax = plt.subplots(2,2,figsize=(10,30))
fig.suptitle("Daytype'a göre Top 20 sıralaması")
sns.barplot(ax=ax[0][0],x=x_a_m, y=y_a_m)
ax[0][0].set_title("Morning")
sns.barplot(ax=ax[0][1],x=x_a_a, y=y_a_a)
ax[0][1].set_title("Afternoon")
sns.barplot(ax=ax[1][0],x=x_a_e, y=y_a_e)
ax[1][0].set_title("Evening")
sns.barplot(ax=ax[1][1],x=x_a_n, y=y_a_n)
ax[1][1].set_title("Night")
plt.show(block=True)

#   Data frame variables converted to Boolean values. Thanks to this conversion, run time will be decreased.

def df_converter (df):
    df = df.applymap(lambda x: True if x>0 else False)
    return df

m = df_converter(morning)
a = df_converter(afternoon)
e = df_converter(evening)
n = df_converter(night)

#   Night is not analysed. Because total selling at night is 14.

#   Apriori is used by choosing min support as 0.01
m_frq_itemsets = apriori(m, min_support=0.01,use_colnames=True)
a_frq_itemsets = apriori(a, min_support=0.01,use_colnames=True)
e_frq_itemsets = apriori(e, min_support=0.01,use_colnames=True)

#   Association rules is  applied and metric is chosen as confidence.

m_rules = association_rules(m_frq_itemsets,metric="confidence",min_threshold=0.1)
a_rules = association_rules(a_frq_itemsets,metric="confidence",min_threshold=0.1)
e_rules = association_rules(e_frq_itemsets,metric="confidence",min_threshold=0.1)

# arl_recommender function is created for association rule.

def arl_recommender(rules_df, product_name, rec_count=2):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_name:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    d = {x:recommendation_list.count(x) for x in recommendation_list}
    liste = list(d.keys())

    return liste[0:rec_count]

#   To ıncrease the selling 3rd item in top 10, 2 items will be recommended for each day type.


arl_recommender(m_rules,"Pastry")
arl_recommender(a_rules,"Tea")
arl_recommender(e_rules,"Tea")