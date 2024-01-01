import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
from streamlit_option_menu import option_menu
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)

data,le_owner,le_brand,rf=pickle.load(open('model.pkl','rb'))


def home():
    ''' FUNCTION TO CREATE HOME PAGE'''
    # WEB-PAGE TITLE
    st.title('Price💲 Prediction of Used Bikes')
        
    # DISPLAYING TEXT
    #st.markdown('** Home Page **')
    
    # DISPLAYING IMAGE
    st.image(r'bike.jpg',use_column_width=True)
    
    # DISPLAYING SOME TEXT
    st.write('''
    Hey Guys!!!🙋‍♂️ Welcome to our Website 🙌🎆😎. \n 
    This website is used to predict the price💲of used bikes.\n
    This website estimates the price💲 of any bike based on Machine Learning Approach. In the Machine Learning approach, we provide the Machine
    learning algorithm with some data. Then the algorithm creates a Machine Learning model which estimates the selling price of used cars
    based on the  features of the bikes. The input to the machine learning model would be given by the end user or the 
    client.\n
    ''')

@st.cache(suppress_st_warning=True)
def scatter():
    sns.lmplot(x='age',y='price',data=data,fit_reg=False,hue='owner',legend=True,palette='Set1')
    st.pyplot()
    
@st.cache(suppress_st_warning=True)
def histogram():
    plt.hist(data['age'],color='green',edgecolor='red',bins=15)
    plt.title('Age VS Frequency')
    plt.xlabel('Age(months)')
    plt.ylabel('Frequency')
    st.pyplot()

@st.cache(suppress_st_warning=True)
def bar():
    sns.countplot(x='brand',data=data)
    plt.xticks(rotation=90)
    st.pyplot()

@st.cache(suppress_st_warning=True)
def box():
    sns.boxplot(x=data['owner'],y=data['price'])
    st.pyplot()

@st.cache(suppress_st_warning=True)
def pair():
    sns.pairplot(data,kind='scatter',hue='owner')
    st.pyplot()

@st.cache(suppress_st_warning=True)
def heat():
    sns.heatmap(data.corr())
    st.pyplot()


def eda():
    st.title("Exploratory Data Analysis of Used Bikes")

    if st.checkbox('Show Data Frame'):
        st.write(data)
        
    if st.checkbox('Show Available Columns'):
        st.write(data.columns)

    if st.checkbox('Description of Numerical Columns'):
        st.write(data.describe())
    
    if st.checkbox('Show Cross table'):
        st.write(pd.crosstab(index=data['brand'],columns=data['owner'],normalize=True,margins=True))

    if st.checkbox('Show Correlation Matrix'):
        st.write(data.corr(method='pearson'))

    if st.checkbox('Show Scatter Plot'):
        scatter()

    if st.checkbox('Show Histogram'):
        histogram()
                     
    if st.checkbox('Show Bar Plot'):
        bar()

    if st.checkbox('Show Box plot'):
        box()

    if st.checkbox('Show Pairwise plot'):
        pair()

    if st.checkbox('Show Heat Map'):
        heat()


@st.cache
def pred(brand,owner,kms_driven,age,power):

    brand=le_brand.transform([brand])

    owner=le_owner.transform([owner])
    
    p=rf.predict([[brand,owner,kms_driven,age,power]])

    st.subheader(f'The price of the bike is ₹{round(np.e**p)}')



@st.cache
def predict():

    st.title("Prediction")

    brand=st.selectbox('Brand',data['brand'].unique())

    owner=st.selectbox('Owner',data['owner'].unique())

    kms_driven=st.slider("Kilometers driven",min_value=round(min(data['kms_driven'])),max_value=round(max(data['kms_driven'])),value=round(data['kms_driven'].mean()),step=1000)

    age=st.slider("Age",min_value=round(min(data['age'])),max_value=round(max(data['age'])),value=round(data['age'].mean()),step=1)

    power=st.slider("Power",min_value=round(min(data['power'])),max_value=round(max(data['power'])),value=round(data['power'].mean()),step=10)

    if st.button("Predict"):
        pred(brand,owner,kms_driven,age,power)

    


choose = option_menu("Main Menu", ["Home", "Exploratory Data Analysis", "Prediction"],
                         icons=['house', 'bar-chart-fill','coin'],
                         menu_icon="cast", default_index=0,
                         orientation='horizontal',
                         
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "yellow", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
        "max-width" : '100%',
                         }
    )

if choose=="Home":
    home()

elif choose=="Exploratory Data Analysis":
    eda()

elif choose=="Prediction":
    predict()