# importing packages
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from reg import get_first

# project description
proj_desc =  '''
    Hey Guys!!!🙋‍♂️ Welcome to our Website 🙌🎆😎. \n This website is used to predict the price💲of used bikes.\nThis website estimates the price💲 of any bike based on Machine Learning Approach. In the Machine Learning approach, we provide the Machine learning algorithm with some data. Then the algorithm creates a Machine Learning model which estimates the selling price of used cars based on the  features of the bikes. The input to the machine learning model would be given by the end user or the client.\n'''

# storing the image location
sell_image = r'predict_page_img3.jpg'

# storing the image location
marketing_image = r'bike.jpg'  

# storing the logo location
logo = r'eda.jpg'

# prediction header
predict_header = 'Predict Selling Price💲 of your car🚗'

# website title
website_title = 'Price💲 Prediction of Used Bikes'

# display class
class Display():

    # defining the constructor
    def __init__(self,choose):

        # home tab
        if choose == 'Home':
            ''' FUNCTION TO CREATE HOME PAGE'''

            # web-page title
            st.title(website_title)
        
            # display text
            #st.markdown('** Home Page **')
            
            # display image
            st.image(marketing_image,use_column_width=True)
            
            # display project description
            st.write(proj_desc)

        # exploratory data analysis tab
        elif choose=="Exploratory Data Analysis":

            # display title
            st.title("Exploratory Data Analysis of Used Bikes")

            # load data
            data=pd.read_csv('Used_Bikes.csv')

            # show data checkbox
            if st.checkbox('Show Data Frame'):
                st.write(data)

            # show columns checkbox    
            if st.checkbox('Show Available Columns'):
                st.write(data.columns)

            # description checkbox    
            if st.checkbox('Description of Numerical Columns'):
                st.write(data.describe())

            # cross table checkbox          
            if st.checkbox('Show Cross table'):
                st.write(pd.crosstab(index=data['brand'],columns=data['owner'],normalize=True,margins=True))

            # correlation checkbox    
            if st.checkbox('Show Correlation Matrix'):
                st.write(data.corr(method='pearson'))

            # scatter plot checkbox    
            if st.checkbox('Show Scatter Plot'):
                sns.lmplot(x='age',y='price',data=data,fit_reg=False,hue='owner',legend=True,palette='Set1')
                st.pyplot()

            # histogram plot checkbox   
            if st.checkbox('Show Histogram'):
                plt.hist(data['age'],color='green',edgecolor='red',bins=15)
                plt.title('Age VS Frequency')
                plt.xlabel('Age(months)')
                plt.ylabel('Frequency')
                st.pyplot()    

            # bar chart checkbox     
            if st.checkbox('Show Bar Plot'):
                sns.countplot(x='brand',data=data)
                plt.xticks(rotation=90)
                st.pyplot()

            # box plot checkbox    
            if st.checkbox('Show Box plot'):
                sns.boxplot(x=data['owner'],y=data['price'])
                #bx.set_ylim(0,0.1)
                st.pyplot()

            # heatmap checkbox          
            if st.checkbox('Show Heat Map'):
                sns.heatmap(data.corr(),annot=True,cmap='Blues')
                st.pyplot()

        # prediction tab    
        elif choose=="Prediction":

            # display title
            st.title("Prediction")
            
            # load data
            data=pd.read_csv('Used_Bikes.csv')

            # remove duplicate rows
            data.drop_duplicates(keep='first',inplace=True)

            # selectbox
            bike_name=st.selectbox('Bike Name',data['bike_name'].unique())
            
            # selectbox
            city=st.selectbox('City',data['city'].unique())

            # selectbox
            owner=st.selectbox('Owner',data['owner'].unique())

            # selectbox
            brand=st.selectbox('Brand',[get_first(bike_name)])

            # slider
            kms_driven=st.slider("Kilometers driven",min_value=round(min(data['kms_driven'])),max_value=round(max(data['kms_driven'])),value=round(data['kms_driven'].mean()),step=1000)

            # slider
            age=st.slider("Age",min_value=round(min(data['age'])),max_value=round(max(data['age'])),value=round(data['age'].mean()),step=1)

            # slider
            power=st.slider("Power",min_value=round(min(data['power'])),max_value=round(max(data['power'])),value=round(data['power'].mean()),step=10)

            # predict button
            if st.button("Predict"):

                # one hot encoding
                data=pd.get_dummies(data,drop_first=True)

                # features
                X=data.drop(['price'],axis='columns',inplace=False)

                # target
                y=data['price']

                # normalize the target
                y=np.log(y)

                # split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

                # random forest regressor
                rf=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=101)

                # fit the model
                model=rf.fit(X_train,y_train)

                # load the data
                data=pd.read_csv('Used_Bikes.csv')

                # drop duplicate rows
                data.drop_duplicates(keep='first',inplace=True)

                # features
                data=data.drop('price',axis='columns',inplace=False)

                # create a dataframe
                d=pd.DataFrame({'bike_name':[bike_name],'city':[city],'kms_driven':[kms_driven],'owner':[owner],'age':[age],'power':[power],'brand':[brand]})

                # add user input
                data=pd.concat([data,d],axis=0,ignore_index=True)

                # one hot encoding
                d=pd.get_dummies(data,drop_first=True)

                # predict the price
                p=rf.predict(d)

                # display the predicted price
                st.subheader(f'The price of the bike is ₹{round(np.e**p[-1])}')



# app title
st.title("Used Bike Price Prediction")

# menu bar
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

# calling display class
Display(choose)

