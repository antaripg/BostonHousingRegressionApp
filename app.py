import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(page_title="Boston Housing Prediction App", page_icon="🏘️")

st.title("Boston Housing Prediction App")
st.markdown("### This app predicts the **Boston House Price**!")
st.write("---")

# Load the Boston Dataset
st.cache_data()
def get_data():
    california = datasets.fetch_california_housing()
    # st.write(california.keys())
    # st.write(california.target_names)
    # california_df = pd.DataFrame(california.data, columns=california.feature_names)
    # st.dataframe(california_df)
    feature_names = california.feature_names
    X = pd.DataFrame(california.data, columns=california.feature_names)
    Y = pd.DataFrame(california.target, columns=["MEDV"])
    return X, Y, feature_names

X, Y, feature_names = get_data()

# Build Regression Model
st.cache_resource()
def call_model(X, Y):
    model = RandomForestRegressor()
    model.fit(X, Y)

model = call_model(X, Y)
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features(feature_names: list):
    feature_data_dict = {}

    for feature_name in feature_names:
        feature_data_dict[feature_name] = st.sidebar.slider(feature_name,
                                                            X[feature_name].min(),
                                                            X[feature_name].max(),
                                                            X[feature_name].mean())
        features = pd.DataFrame(feature_data_dict, index=[0])
    return features

df = user_input_features(feature_names=feature_names)

# Main Panel
# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')


# Apply Model to Make Prediction
# if df != None:
#     prediction = model.predict(df)

#     st.header('Prediction of MEDV')
#     st.write(prediction)
#     st.write('---')

# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')



