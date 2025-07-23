import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Apply seaborn theme for consistent visuals
sns.set_theme(style="whitegrid")

# Load the tips dataset
tips = sns.load_dataset("tips")

# streamlit app title and description¶
st.title('pooja sharma tips data visualization app')
st.write("This is a simple app to visualize the tips dataset using seaborn.")
st.markdown(
    "This is interactive  app showcase  various **seaborn plots** "
    "built on the classic 'tips' dataset. select a plot  from the dropdown "
    "to visualize restaurant tipping trends "
)

# plotting functions(one for each)
# function create and display plot

def display_plot(title, plot_func):

    st.subheader(title)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_func(ax=axis)
    st.pyplot(fig)
    plt.close(fig)

# plot

def scatter_plot(axis):
   sns.scatterplot(data=tips, x="total_bill", y="tip",hue="time", size="size",palette="deep", ax=axis)
   axis.set_title("scatter plot: Scatter plot of total bill vs tip")
   
def line_plot(axis):
    sns.lineplot(data=tips, x= 'size', y='total_bill', hue='sex',markers='o',ax=axis)
    axis.set_title("Line plot of total bill vs tip")

def bar_plot(axis):
    sns.barplot(data=tips, x='day', y='total_bill', hue = 'sex',palette='muted',ax=axis)
    axis.set_title("Barplot of Total Bill by Day")  

def box_plot(axis):
    sns.boxplot(data=tips, x='day', y='tip', hue='smoker', palette='Set2',ax=axis)
    axis.set_title("Boxplot of Tips by Day and Smoker Status")

def violin_plot(axis):
    sns.violinplot(data=tips, x='day', y='total_bill', hue='time', split=True, palette="pastel", ax=axis)
    axis.set_title("Violin Plot of Total Bill by Day and Time")

def count_plot(axis):
    sns.countplot(data=tips, x='day', hue='smoker', palette='dark', ax=axis)
    axis.set_title("Count Plot of Days by Smoker Status")

def reg_plot(axis):
    sns.regplot(data=tips, x='total_bill', y='tip', scatter_kws={'s':50}, line_kws={'color':'red'},ax=axis)
    axis.set_title("Regression Plot of Total Bill vs Tip")

def hist_plot(axis):
    sns.histplot(data=tips, x='total_bill', bins=20, kde=True, color='blue',ax=axis)
    axis.set_title("Histogram of Total Bill with KDE")

def strip_plot(axis):
    sns.stripplot(data=tips, x='day', y='tip', hue='sex', jitter=True, palette='set1',ax=axis)
    axis.set_title("strip plot: Tips by data and gender")

def kde_plot(axis):
    sns.kdeplot(data=tips, x='total_bill',hue='sex', fill=True, palette='tab10',ax=axis)
    axis.set_title("kde plot:Total bill density by gender")
    
# dictionary to map plot names to functions
plot_options = {
    "Scatter Plot" : scatter_plot,
    "Line Plot" : line_plot,
    "Bar Plot" : bar_plot,
    "Box Plot" : box_plot,
    "Violin Plot" : violin_plot,
    "Count Plot" : count_plot,
    "Count Plot": count_plot,
    "Regression Plot" : reg_plot,    
    "Histogram Plot" : hist_plot,
    "Strip Plot" : strip_plot,
    "KDE Plot" : kde_plot
}

# streamlit UI : select and render plot
selected_plot = st.selectbox(" select a plot to visualize :", list(plot_options.keys()))

# display selected plot
st.subheader(f"{selected_plot}")
fig,axis= plt.subplots(figsize=(6,4.5))
plot_function=plot_options[selected_plot]
plot_function(axis)
st.pyplot(fig)
plt.close(fig)

# footer(optional)
st.markdown("---")
st.caption("made with love using seaborn & streamlit | Pooja sharma")