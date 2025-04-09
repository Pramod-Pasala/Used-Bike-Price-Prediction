# importing packages
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

model = joblib.load('model.pkl')
le_brand = joblib.load('le_brand.pkl')
le_owner = joblib.load('le_owner.pkl')

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
            st.image(marketing_image,use_container_width=True)
            
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
                numeric_data = data.select_dtypes(include=['number'])
                st.write(numeric_data.corr(method='pearson'))

            # scatter plot checkbox    
            if st.checkbox('Show Scatter Plot'):
                fig = self.generate_plot(
                    data,
                    plot_type='scatter',
                    x_col='age',
                    y_col='price',
                    hue_col='owner',
                    palette='Set1',
                    legend=True,
                    title='Age vs Price by Owner'
                )
                st.pyplot(fig)

            # histogram plot checkbox   
            if st.checkbox('Show Histogram'):
                fig = self.generate_plot(
                    data,
                    plot_type='histogram',
                    x_col='age',
                    color='green',
                    edgecolor='red',
                    bins=15,
                    title='Age VS Frequency',
                    xlabel='Age(months)',
                    ylabel='Frequency'
                )
                st.pyplot(fig)   

            # bar chart checkbox     
            if st.checkbox('Show Bar Plot'):
                fig = self.generate_plot(
                    data,
                    plot_type='bar',
                    x_col='brand',
                    xtick_rotation=90,
                    title='Count of Items by Brand'
                )
                st.pyplot(fig)

            # box plot checkbox    
            if st.checkbox('Show Box plot'):
                fig = self.generate_plot(
                    data,
                    plot_type='box',
                    x_col='owner',
                    y_col='price',
                    title='Price Distribution by Owner'
                )
                st.pyplot(fig)

            # heatmap checkbox          
            if st.checkbox('Show Heat Map'):
                fig = self.generate_plot(
                    data.select_dtypes(include=['number']),
                    plot_type='heatmap',
                    annot=True,
                    cmap='Blues',
                    title='Correlation Heatmap',
                    figsize=(10, 8)  # Slightly larger for readability
                )
                st.pyplot(fig)
        # prediction tab    
        elif choose=="Prediction":

            # display title
            st.title("Prediction")
            
            # load data
            data=pd.read_csv('Used_Bikes.csv')

            # remove duplicate rows
            data.drop_duplicates(keep='first',inplace=True)

            # selectbox
            brand=st.selectbox('Brand',data['brand'].unique())
            brand=le_brand.transform([brand])[0]

            # selectbox
            owner=st.selectbox('Owner',data['owner'].unique())
            owner=le_owner.transform([owner])[0]

            # slider
            kms_driven=st.slider("Kilometers driven",min_value=round(min(data['kms_driven'])),max_value=round(max(data['kms_driven'])),value=round(data['kms_driven'].mean()),step=1000)

            # slider
            age=st.slider("Age",min_value=round(min(data['age'])),max_value=round(max(data['age'])),value=round(data['age'].mean()),step=1)

            # slider
            power=st.slider("Power",min_value=round(min(data['power'])),max_value=round(max(data['power'])),value=round(data['power'].mean()),step=10)

            # predict button
            if st.button("Predict"):
                feature_names = ['kms_driven', 'owner', 'age', 'power', 'brand']
                input_data = pd.DataFrame([[kms_driven, owner, age, power, brand]], columns=feature_names)
                prediction = model.predict(input_data)
                prediction = np.exp(prediction)
                # display the predicted price
                st.subheader(f'The price of the bike is ₹{int(prediction[0])}')

    @st.cache_data
    def generate_plot(_self,data,plot_type,**kwargs):

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
        
        if plot_type == 'scatter':
            # Scatter plot
            sns.scatterplot(
                x=kwargs.get('x_col', 'age'),
                y=kwargs.get('y_col', 'price'),
                data=data,
                hue=kwargs.get('hue_col', 'owner'),
                palette=kwargs.get('palette', 'Set1'),
                legend=kwargs.get('legend', True),
                ax=ax
            )
        
        elif plot_type == 'histogram':
            # Histogram
            ax.hist(
                data[kwargs.get('x_col', 'age')],
                color=kwargs.get('color', 'green'),
                edgecolor=kwargs.get('edgecolor', 'red'),
                bins=kwargs.get('bins', 15)
            )
            ax.set_title(kwargs.get('title', 'Age VS Frequency'))
            ax.set_xlabel(kwargs.get('xlabel', 'Age(months)'))
            ax.set_ylabel(kwargs.get('ylabel', 'Frequency'))
        
        elif plot_type == 'bar':
            # Bar plot (countplot)
            sns.countplot(
                x=kwargs.get('x_col', 'brand'),
                data=data,
                ax=ax
            )
            ax.tick_params(axis='x', rotation=kwargs.get('xtick_rotation', 90))
        
        elif plot_type == 'box':
            # Box plot
            sns.boxplot(
                x=kwargs.get('x_col', 'owner'),
                y=kwargs.get('y_col', 'price'),
                data=data,
                ax=ax
            )
            
        elif plot_type == 'heatmap':
            # Heatmap
            sns.heatmap(
                data.corr(),
                annot=kwargs.get('annot', True),
                cmap=kwargs.get('cmap', 'Blues'),
                ax=ax
            )
        
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")
        
   
        if 'title' in kwargs and plot_type != 'histogram':  
            ax.set_title(kwargs['title'])
        
        return fig  


# app title
st.title("Used Bike Price Prediction")

# menu bar
choose = option_menu(
    "Main Menu",
    ["Home", "Exploratory Data Analysis", "Prediction"],
    icons=['house', 'bar-chart-fill','coin'],
    menu_icon="cast",
    default_index=0,
    orientation='horizontal',
    styles={
        "container": {
            "padding": "5!important", 
            "background-color": "#ffffff",  # Pure white background
            "max-width": "100%",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.1)"
        },
        "icon": {
            "color": "orange",  # Icons in bright color
            "font-size": "25px"
        }, 
        "nav-link": {
            "font-size": "16px", 
            "text-align": "left", 
            "margin": "0px", 
            "--hover-color": "#f2f2f2",
            "color": "black"  # Text for unselected tabs
        },
        "nav-link-selected": {
            "background-color": "#02ab21",  # Green highlight
            "color": "white"  # Selected tab text
        }
    }
)

# calling display class
Display(choose)

