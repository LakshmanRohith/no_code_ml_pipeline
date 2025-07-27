import streamlit as st
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB # For Naive Bayes
from sklearn.neural_network import MLPClassifier, MLPRegressor # For ANN
import xgboost as xgb # For XGBoost
import lightgbm as lgb # For LightGBM
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, roc_curve, auc
)
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time
import numpy as np # For numerical operations, especially for ROC curve
import pandas.api.types as pd_types # For robust type checking

# --- IMPORTANT: Ensure these libraries are installed ---
# pip install category_encoders xgboost lightgbm

# --- Main App Logic ---

# --- Callback for Phase 6 to Phase 7 transition ---
def next_to_model_training_callback():
    print(f"*** DEBUG (Terminal): next_to_model_training_callback executed. Current phase before update: {st.session_state.current_phase}")
    st.session_state.current_phase = 6
    print(f"*** DEBUG (Terminal): current_phase after callback: {st.session_state.current_phase}")

# --- Callback for Phase 7 to Phase 8 transition ---
def next_to_model_evaluation_callback():
    print(f"*** DEBUG (Terminal): next_to_model_evaluation_callback executed. Current phase before update: {st.session_state.current_phase}")
    st.session_state.current_phase = 7
    print(f"*** DEBUG (Terminal): current_phase after callback: {st.session_state.current_phase}")

# --- App title and header ---
st.set_page_config(page_title="No-Code ML Pipeline", layout="wide")
st.title("🚀 No-Code Machine Learning Pipeline")

# --- Initialize session state for phase control and data ---
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = 0
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {} # To store trained models and their metrics

# --- DEBUG: Display current phase at the top of the sidebar ---
st.sidebar.write(f"**DEBUG: Current Phase (Global):** {st.session_state.get('current_phase', 'Not Set')}")

# --- Create folders if not exist ---
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- PHASE 1: UPLOAD DATA ---
if st.session_state.current_phase == 0:
    st.markdown("### Step 1: Upload your dataset (CSV format only)")

    uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV dataset", type=["csv"])

    if uploaded_file is not None:
        file_path = os.path.join("data", f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ File uploaded and saved as {os.path.basename(file_path)}")

        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df.copy()
            st.session_state.file_uploaded = True

            st.markdown("### 📊 Dataset Preview (First 5 Rows)")
            st.dataframe(st.session_state.df.head())

            st.markdown("### ℹ️ Dataset Info")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Number of Rows:**", st.session_state.df.shape[0])
                st.write("**Number of Columns:**", st.session_state.df.shape[1])
                st.write("**Column Names:**", list(st.session_state.df.columns))

            with col2:
                st.write("**Data Types:**")
                st.write(st.session_state.df.dtypes)
                st.write("**Missing Values:**")
                st.write(st.session_state.df.isnull().sum())

            if st.button("Next Phase: Data Visualization ➡️"):
                st.session_state.current_phase = 1
                st.rerun()

        except Exception as e:
            st.error(f"❌ Error reading the file: {e}")
            st.session_state.file_uploaded = False
            st.session_state.df = None
    else:
        st.warning("👈 Please upload a .csv file to begin.")

# --- PHASE 2: DATA VISUALIZATION ---
if st.session_state.current_phase == 1 and st.session_state.file_uploaded:
    st.markdown("---")
    st.markdown("### 📊 Step 2: Explore and Visualize Your Data")

    df = st.session_state.df.copy() # Use a copy to avoid modifying original df accidentally

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    all_cols = df.columns.tolist()

    st.sidebar.markdown("### 📌 Visualization Options")
    chart_type = st.sidebar.radio("Select Chart Type", ["Univariate", "Bivariate"], key="chart_type_radio")

    if chart_type == "Univariate":
        st.subheader("Single Column Visualizations")
        selected_col = st.sidebar.selectbox("Select a column to visualize", all_cols, key="viz_select_uni_col")

        if selected_col:
            col_type = 'Numerical' if selected_col in num_cols else 'Categorical'

            st.markdown(f"#### Column Selected: **{selected_col}** ({col_type})")

            if col_type == 'Numerical':
                st.subheader("📈 Histogram")
                fig = px.histogram(df, x=selected_col, marginal="box", hover_data=df.columns)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📦 Box Plot")
                fig = px.box(df, y=selected_col)
                st.plotly_chart(fig, use_container_width=True)

            elif col_type == 'Categorical':
                st.subheader("📊 Value Counts (Bar Chart)")
                fig = px.bar(df[selected_col].value_counts().reset_index(),
                             x='index', y=selected_col,
                             labels={'index': selected_col, selected_col: 'Count'},
                             title=f"Value Counts of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("🥧 Pie Chart")
                pie_data = df[selected_col].value_counts().reset_index()
                pie_data.columns = ['Category', 'Count']
                fig = px.pie(pie_data, values='Count', names='Category', title=f"Pie Chart of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bivariate":
        st.subheader("Two-Column Visualizations")
        col_x, col_y = st.sidebar.columns(2)
        selected_col_x = col_x.selectbox("Select X-axis column", all_cols, key="viz_select_bi_x_col")
        selected_col_y = col_y.selectbox("Select Y-axis column", all_cols, key="viz_select_bi_y_col")

        if selected_col_x and selected_col_y:
            st.markdown(f"#### X: **{selected_col_x}** vs. Y: **{selected_col_y}**")

            x_type = 'Numerical' if selected_col_x in num_cols else 'Categorical'
            y_type = 'Numerical' if selected_col_y in num_cols else 'Categorical'

            # Numerical vs Numerical
            if x_type == 'Numerical' and y_type == 'Numerical':
                st.subheader("✨ Scatter Plot")
                fig = px.scatter(df, x=selected_col_x, y=selected_col_y,
                                 title=f"Scatter Plot of {selected_col_x} vs {selected_col_y}",
                                 hover_data=df.columns)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📈 Line Plot (if applicable, e.g., time series)")
                try:
                    # Try to plot line if X-axis can be sorted meaningfully
                    df_sorted = df.sort_values(by=selected_col_x)
                    fig_line = px.line(df_sorted, x=selected_col_x, y=selected_col_y,
                                       title=f"Line Plot of {selected_col_x} vs {selected_col_y}",
                                       hover_data=df_sorted.columns)
                    st.plotly_chart(fig_line, use_container_width=True)
                except Exception:
                    st.info("Line plot might not be suitable for the selected columns or data is not sortable.")

            # Categorical vs Numerical
            elif x_type == 'Categorical' and y_type == 'Numerical':
                st.subheader("📊 Box Plot (Numerical by Category)")
                fig = px.box(df, x=selected_col_x, y=selected_col_y,
                             title=f"Box Plot of {selected_col_y} by {selected_col_x}")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📈 Bar Plot (Average Numerical by Category)")
                fig = px.bar(df.groupby(selected_col_x)[selected_col_y].mean().reset_index(),
                             x=selected_col_x, y=selected_col_y,
                             title=f"Average {selected_col_y} by {selected_col_x}")
                st.plotly_chart(fig, use_container_width=True)

            # Numerical vs Categorical (swap roles for consistent plotting)
            elif x_type == 'Numerical' and y_type == 'Categorical':
                st.subheader("📊 Box Plot (Numerical by Category)")
                fig = px.box(df, x=selected_col_y, y=selected_col_x,
                             title=f"Box Plot of {selected_col_x} by {selected_col_y}")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📈 Bar Plot (Average Numerical by Category)")
                fig = px.bar(df.groupby(selected_col_y)[selected_col_x].mean().reset_index(),
                             x=selected_col_y, y=selected_col_x,
                             title=f"Average {selected_col_x} by {selected_col_y}")
                st.plotly_chart(fig, use_container_width=True)

            # Categorical vs Categorical
            elif x_type == 'Categorical' and y_type == 'Categorical':
                st.subheader("📊 Count Plot (Stacked Bar Chart)")
                counts_df = df.groupby([selected_col_x, selected_col_y]).size().reset_index(name='Count')
                fig = px.bar(counts_df, x=selected_col_x, y='Count', color=selected_col_y,
                             title=f"Count of {selected_col_x} by {selected_col_y}",
                             labels={'Count': 'Number of Occurrences'})
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("🔥 Heatmap of Counts")
                cross_tab = pd.crosstab(df[selected_col_x], df[selected_col_y])
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f"Heatmap of Counts: {selected_col_x} vs {selected_col_y}")
                st.pyplot(fig)

    if len(num_cols) >= 2:
        st.markdown("### 🔥 Correlation Heatmap (All Numerical Columns)")
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    if st.button("Next Phase: Data Cleaning ➡️"):
        st.session_state.current_phase = 2
        st.rerun()

# --- PHASE 3: DATA CLEANING ---
if st.session_state.current_phase == 2 and st.session_state.file_uploaded:
    st.markdown("---")
    st.markdown("### 🧹 Step 3: Clean Your Data")

    df = st.session_state.df # Work directly on the session state df

    st.subheader("🔍 Missing Value Handling")
    missing_df = df.isnull().sum()
    missing_df = missing_df[missing_df > 0]

    missing_col1, missing_col2 = st.columns(2)

    with missing_col1:
        st.markdown("#### Missing Value Summary")
        if missing_df.empty:
            st.success("✅ No missing values detected in the dataset.")
        else:
            st.warning("⚠️ Missing values detected. Please address them below.")
            missing_info_table = pd.DataFrame({
                'Column Name': missing_df.index,
                'Data Type': df[missing_df.index].dtypes.values,
                'Missing Count': missing_df.values,
                'Missing Percentage': (missing_df.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_info_table, use_container_width=True)
            
    with missing_col2:
        st.markdown("#### Choose Handling Method")
        if not missing_df.empty:
            imputation_choices = {}
            for col in missing_df.index:
                col_type = df[col].dtype
                
                options = ["None (Leave as is)", "Drop Column"]
                if pd_types.is_numeric_dtype(col_type):
                    options.insert(1, "Fill with Mean")
                    options.insert(2, "Fill with Median")
                options.insert(3, "Fill with Mode") # Mode is always an option

                imputation_choices[col] = st.selectbox(
                    f"Column '{col}' ({col_type}, {missing_df[col]} missing)",
                    options,
                    key=f"missing_impute_{col}"
                )

            if st.button("Apply Missing Value Handling", key="apply_missing_button"):
                for col, method in imputation_choices.items():
                    if method == "Drop Column":
                        df.drop(columns=[col], inplace=True)
                        st.success(f"🗑️ Dropped column '{col}'.")
                    elif method == "Fill with Mean":
                        if pd_types.is_numeric_dtype(df[col]):
                            df[col].fillna(df[col].mean(), inplace=True)
                            st.success(f"📊 Filled missing in '{col}' with its mean.")
                        else:
                            st.error(f"❌ Cannot fill non-numeric column '{col}' with mean. Please choose another method.")
                    elif method == "Fill with Median":
                        if pd_types.is_numeric_dtype(df[col]):
                            df[col].fillna(df[col].median(), inplace=True)
                            st.success(f"📈 Filled missing in '{col}' with its median.")
                        else:
                            st.error(f"❌ Cannot fill non-numeric column '{col}' with median. Please choose another method.")
                    elif method == "Fill with Mode":
                        mode_val = df[col].mode()
                        if not mode_val.empty:
                            df[col].fillna(mode_val[0], inplace=True)
                            st.success(f"🏷️ Filled missing in '{col}' with its mode.")
                        else:
                            st.warning(f"❗ Could not determine mode for '{col}'. Skipping.")
                st.session_state.df = df.copy() # Update session state after modifications
                st.toast("Missing value handling applied! Dataset updated.", icon="✅")
                time.sleep(1) # Give time for toast to show
                st.rerun()
        else:
            st.info("No missing values to handle.")

    st.markdown("---")
    st.subheader("📋 Duplicate Rows Handling")
    dup_count = df.duplicated().sum()

    dup_col1, dup_col2 = st.columns(2)

    with dup_col1:
        st.markdown("#### Duplicate Row Summary")
        if dup_count == 0:
            st.success("✅ No duplicate rows found in the dataset.")
        else:
            st.warning(f"⚠️ {dup_count} duplicate rows detected.")
            
    with dup_col2:
        if dup_count > 0:
            if st.button("Remove Duplicate Rows", key="remove_dups_button"):
                df.drop_duplicates(inplace=True)
                st.session_state.df = df.copy()
                st.success("🗑️ Successfully removed duplicate rows.")
                st.toast("Duplicate rows removed!", icon="✅")
                time.sleep(1)
                st.rerun()
        else:
            st.info("No duplicate rows to remove.")

    st.markdown("---")
    st.subheader("✨ Cleaned Dataset Preview")
    st.dataframe(st.session_state.df.head(), use_container_width=True)
    st.write(f"Current number of rows after cleaning: **{st.session_state.df.shape[0]}**")


    if st.button("Next Phase: Data Visualization ➡️"): # Changed to visualization as we removed definitions
        st.session_state.current_phase = 3 # Go back to feature engineering, the next actual phase.
        st.rerun()

# --- PHASE 4: FEATURE ENGINEERING ---
if st.session_state.current_phase == 3 and st.session_state.file_uploaded:
    st.markdown("---")
    st.markdown("### 🧠 Step 4: Feature Engineering")

    df = st.session_state.df # Work directly on the session state df

    # --- Drop Unwanted Columns ---
    st.container(border=True).markdown("#### 🧹 Drop Unwanted Columns")
    drop_cols_initial = st.multiselect("Select columns to drop (Manual Selection)", df.columns.tolist(), key="drop_cols_select_initial")
    if st.button("Drop Selected Columns (Manual)", key="drop_cols_button_initial"):
        if drop_cols_initial:
            df.drop(columns=drop_cols_initial, inplace=True)
            st.session_state.df = df.copy()
            st.success(f"🗑️ Dropped manually selected columns: {drop_cols_initial}")
            st.rerun()
        else:
            st.warning("Please select columns to drop.")

    st.markdown("---") # Separator

    # --- Recommendation: Drop Unique Identifier Columns ---
    st.container(border=True).markdown("#### 💡 Recommendation: Drop Unique Identifier Columns")
    unique_id_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    if unique_id_cols:
        st.info(f"The following columns have all unique values and are likely identifiers (e.g., 'ID', 'Name'). It's generally recommended to drop them as they don't provide predictive power for most models: **{', '.join(unique_id_cols)}**")
        if st.button(f"Drop Recommended Unique Columns ({len(unique_id_cols)})", key="drop_unique_cols_button"):
            df.drop(columns=unique_id_cols, inplace=True)
            st.session_state.df = df.copy()
            st.success(f"🗑️ Dropped unique identifier columns: {unique_id_cols}")
            st.rerun()
    else:
        st.success("✅ No columns found with all unique values to recommend dropping.")

    st.markdown("---") # Separator

    # --- Categorical Encoding ---
    st.container(border=True).markdown("#### 🔤 Categorical Encoding (Per Column)")
    
    # Get current target column name from session state
    current_target_col_name = st.session_state.get('target_col', None)
    
    # Filter out the target column from categorical columns available for encoding
    all_cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols_for_encoding = [col for col in all_cat_cols if col != current_target_col_name]

    if current_target_col_name and current_target_col_name in all_cat_cols:
        st.info(f"💡 Note: The selected target column ('**{current_target_col_name}**') is categorical and will be automatically handled in Phase 5. It is **not** available for encoding here as a feature.")
        st.markdown("---") # Separator after the note

    if cat_cols_for_encoding:
        st.write("Categorical columns detected for feature encoding:", cat_cols_for_encoding)

        st.markdown("Please select an encoding method for each categorical column:")
        # Simplified encoding options
        encoding_options = ["None", "Label Encoding", "One-Hot Encoding"] 
        
        # Initialize or update encoding choices in session state
        if 'encoding_choices_per_col_state' not in st.session_state:
            st.session_state.encoding_choices_per_col_state = {col: "None" for col in cat_cols_for_encoding}
        
        # Ensure session state reflects current categorical columns (after filtering target)
        for col in cat_cols_for_encoding:
            if col not in st.session_state.encoding_choices_per_col_state:
                st.session_state.encoding_choices_per_col_state[col] = "None"
        cols_to_remove = [col for col in st.session_state.encoding_choices_per_col_state if col not in cat_cols_for_encoding]
        for col in cols_to_remove:
            del st.session_state.encoding_choices_per_col_state[col]

        encoding_choices_to_apply = {}
        for col in cat_cols_for_encoding:
            current_choice = st.session_state.encoding_choices_per_col_state[col]
            
            col1_e, col2_e = st.columns([0.6, 0.4]) # Split for selectbox and info

            with col1_e:
                selected_method = st.selectbox(
                    f"Encoding for '{col}'",
                    encoding_options,
                    index=encoding_options.index(current_choice),
                    key=f"encoding_method_for_{col}"
                )
                encoding_choices_to_apply[col] = selected_method
                st.session_state.encoding_choices_per_col_state[col] = selected_method # Update session state immediately

            with col2_e:
                # Provide recommendation based on column characteristics (simplified for Label/One-Hot)
                n_unique = df[col].nunique()
                if n_unique <= 2:
                    st.info("💡 **Recommendation:** `Label Encoding` (Simple for binary categories).")
                else: # For >2 unique values, One-Hot is generally safe for nominal data
                    st.info("💡 **Recommendation:** `One-Hot Encoding` (Good for nominal data, avoids implied order).")

        if st.button("Apply Selected Encodings", key="apply_individual_encoding_button"):
            applied_encodings = []
            temp_df = df.copy() # Work on a temporary copy for encoding
            
            # Temporarily separate the target column if it exists in the main df
            y_original_target = None
            if current_target_col_name and current_target_col_name in temp_df.columns:
                y_original_target = temp_df[current_target_col_name]
                temp_df = temp_df.drop(columns=[current_target_col_name])

            for col in cat_cols_for_encoding:
                method = encoding_choices_to_apply[col]
                if method != "None" and col in temp_df.columns: # Ensure column still exists in temp_df
                    try:
                        if method == "Label Encoding":
                            encoder = LabelEncoder()
                            temp_df[col] = encoder.fit_transform(temp_df[col].astype(str))
                            applied_encodings.append(f"'{col}' with {method}")
                        elif method == "One-Hot Encoding":
                            temp_df = pd.get_dummies(temp_df, columns=[col], prefix=col)
                            applied_encodings.append(f"'{col}' with {method}")
                        
                    except Exception as e:
                        st.error(f"❌ Encoding failed for column '{col}' with {method}: {e}")
            
            # Re-add the target column to the DataFrame if it was temporarily separated
            if y_original_target is not None:
                temp_df[current_target_col_name] = y_original_target.reset_index(drop=True)


            st.session_state.df = temp_df.copy() # Update session state with the new DataFrame
            if applied_encodings:
                st.success(f"✅ Applied encoding to columns: {', '.join(applied_encodings)}.")
                st.toast("Categorical encoding applied!", icon="✅")
            else:
                st.info("No encoding methods were selected or applied.")
            time.sleep(1)
            st.rerun()

    else:
        st.info("✅ No categorical columns found for feature encoding (or they are the selected target column).")
    
    st.markdown("---") # Separator for next section

    # --- Feature Scaling ---
    st.container(border=True).markdown("#### 📏 Feature Scaling (Per Column)")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Filter out the target column if it's numerical
    numerical_cols_for_scaling = [col for col in num_cols if col != current_target_col_name]

    if current_target_col_name and current_target_col_name in num_cols:
        st.info(f"💡 Note: The selected target column ('**{current_target_col_name}**') is numerical and will not be scaled here as a feature. Scaling of the target is usually not required.")
        st.markdown("---") # Separator after the note


    if numerical_cols_for_scaling:
        st.write("Numerical columns detected for scaling:", numerical_cols_for_scaling)

        st.markdown("Please select a scaling method for each numerical column:")
        # Simplified scaling options
        scaling_options = ["None", "StandardScaler", "MinMaxScaler"]
        
        # Initialize or update scaling choices in session state
        if 'scaling_choices_per_col_state' not in st.session_state:
            st.session_state.scaling_choices_per_col_state = {col: "None" for col in numerical_cols_for_scaling}
        
        # Ensure session state reflects current numerical columns (after filtering target)
        for col in numerical_cols_for_scaling:
            if col not in st.session_state.scaling_choices_per_col_state:
                st.session_state.scaling_choices_per_col_state[col] = "None"
        cols_to_remove_scale = [col for col in st.session_state.scaling_choices_per_col_state if col not in numerical_cols_for_scaling]
        for col in cols_to_remove_scale:
            del st.session_state.scaling_choices_per_col_state[col]

        # Display selectboxes and recommendations for each column
        scaling_choices_to_apply = {}
        for col in numerical_cols_for_scaling:
            current_choice = st.session_state.scaling_choices_per_col_state[col]
            
            col1_s, col2_s = st.columns([0.6, 0.4]) # Split for selectbox and info

            with col1_s:
                selected_method = st.selectbox(
                    f"Scaling for '{col}'",
                    scaling_options,
                    index=scaling_options.index(current_choice),
                    key=f"scaling_method_for_{col}"
                )
                scaling_choices_to_apply[col] = selected_method
                st.session_state.scaling_choices_per_col_state[col] = selected_method # Update session state immediately

            with col2_s:
                # Provide recommendation based on column characteristics (simplified for MinMax/Standard)
                if df[col].nunique() <= 2: # Binary or constant column
                    st.info("💡 **Recommendation:** `None` (Binary/Constant column, scaling not needed).")
                else:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    has_outliers = (df[col] < lower_bound).any() or (df[col] > upper_bound).any()

                    if has_outliers:
                        st.info("💡 **Recommendation:** `StandardScaler` (Robust to outliers, centers data).")
                    else:
                        st.info("💡 **Recommendation:** `MinMaxScaler` (No significant outliers, scales to 0-1 range).")

        if st.button("Apply Selected Scaling Methods", key="apply_individual_scaling_button"):
            applied_scalings = []
            temp_df = df.copy() # Work on a temporary copy for encoding

            # Temporarily separate the target column if it exists in the main df
            y_original_target = None
            if current_target_col_name and current_target_col_name in temp_df.columns:
                y_original_target = temp_df[current_target_col_name]
                temp_df = temp_df.drop(columns=[current_target_col_name])


            for col, method in scaling_choices_to_apply.items():
                if method != "None" and col in temp_df.columns: # Ensure column still exists in temp_df
                    scaler = None
                    if method == "StandardScaler":
                        scaler = StandardScaler()
                    elif method == "MinMaxScaler":
                        scaler = MinMaxScaler()

                    if scaler:
                        try:
                            # Scalers expect 2D array, so pass df[[col]]
                            temp_df[col] = scaler.fit_transform(temp_df[[col]])
                            applied_scalings.append(f"'{col}' with {method}")
                        except Exception as e:
                            st.error(f"❌ Scaling failed for column '{col}' with {method}: {e}")
            
            # Re-add the target column to the DataFrame if it was temporarily separated
            if y_original_target is not None:
                temp_df[current_target_col_name] = y_original_target.reset_index(drop=True)


            st.session_state.df = temp_df.copy()
            if applied_scalings:
                st.success(f"✅ Applied scaling to columns: {', '.join(applied_scalings)}.")
                st.toast("Feature scaling applied!", icon="✅")
            else:
                st.info("No scaling methods were selected or applied.")
            time.sleep(1)
            st.rerun()

    else:
        st.info("✅ No numerical columns found for feature scaling.")
    
    st.markdown("---") # Separator for next section

    # --- Understanding Column Types Guide ---
    st.container(border=True).markdown("#### 📚 Understanding Column Types: A Guide")
    st.markdown("""
    When preparing your data for machine learning, it's crucial to understand the type of information each column holds. This guide helps you classify your columns and decide how to process them.
    """)
    
    with st.expander("Expand to learn about Column Types"):
        st.markdown("""
        **✅ Step 1: Check Data Type**
        Use the `.dtypes` function in Pandas (already shown in Phase 1).
        * If the data type is: `int64` or `float64` → Likely numerical
        * If `object` or `string` → Could be categorical, text, or ID

        **✅ Step 2: Check Number of Unique Values**
        Use `.nunique()` to count unique values.
        * If almost every row has a unique value (e.g., `nunique() == len(df)`) → It's likely an **identifier** or **free-form text**.
        * If there are only a few unique values (e.g., < 15-20, depending on dataset size) → It's likely **categorical** or **ordinal**.
        * If many unique numbers → Likely **numerical**.

        **✅ Step 3: Inspect Sample Values**
        Use `.unique()` to print a few values from the column (you can do this in Phase 1 or 2 by selecting columns).
        * If values are like "male", "female" → **Categorical**
        * If values are like "Low", "Medium", "High" → **Ordinal**
        * If values are names, sentences, or unique codes → **Text** or **Identifier**
        * If values are integers or floats that represent a quantity → **Numerical**

        **✅ Step 4: Use Domain Knowledge**
        Think about what the column represents in the real world:
        * Does it have an inherent order (e.g., 'small', 'medium', 'large')? → Possibly **Ordinal**
        * Does it describe a name, ID, or a long description? → **Text** or **Identifier**
        * Is it a quantity you can perform mathematical operations on (e.g., age, price, temperature)? → **Numerical**
        * Is it a category without any natural order (e.g., 'red', 'blue', 'green')? → **Categorical**

        **✅ Step 5: Decide How to Process the Column**
        Based on the above, here's how you might typically process them in this pipeline:
        * **Numerical** → Keep, and possibly scale (using the section above).
        * **Categorical** → Encode using Label or One-Hot Encoding (using the section above).
        * **Ordinal** → Encode using Label Encoding (as it preserves order, unlike one-hot for this purpose).
        * **Text / Free-form** → Often dropped in this basic pipeline (use "Drop Unwanted Columns"). For advanced use, would require Natural Language Processing (NLP).
        * **Identifier** → Drop (using the "Drop Unique Identifier Columns" recommendation or "Drop Unwanted Columns"). Not useful for modeling.
        """)
    
    st.markdown("---") # Separator for next section

    st.subheader("✅ Updated Dataset Preview (First 5 Rows)")
    st.dataframe(st.session_state.df.head(), use_container_width=True)
    st.write(f"Current number of rows: **{st.session_state.df.shape[0]}**")
    st.write(f"Current number of columns: **{st.session_state.df.shape[1]}**")


    if st.button("Next Phase: Target Selection ➡️"):
        st.session_state.current_phase = 4
        st.rerun()

# --- PHASE 5: TARGET SELECTION ---
if st.session_state.current_phase == 4 and st.session_state.file_uploaded:
    st.markdown("---")
    st.markdown("### 🎯 Step 5: Select Target Column & Detect ML Task")

    df = st.session_state.df

    st.container(border=True).markdown("#### Target Column Selection")
    target_col = st.selectbox("🟢 Select the target column (label)", df.columns.tolist(), key="target_col_select")
    
    if target_col:
        target_unique = df[target_col].nunique()
        target_dtype = df[target_col].dtype

        ml_type = ""
        # Check if column is numeric and has more than a certain number of unique values for regression
        # Using a threshold of 15 unique values is a common heuristic for distinguishing regression from classification for numerical targets.
        if pd_types.is_numeric_dtype(target_dtype) and target_unique > 15:
            ml_type = "Regression"
        # If it's an object type (string) or has few unique values (even if numeric, could be discrete classes)
        elif pd_types.is_object_dtype(target_dtype) or target_unique <= 15:
            ml_type = "Classification"
        else:
            ml_type = "Could not determine (check data type/unique values)"

        try:
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # --- IMPORTANT: Auto-Label Encode Categorical Target ---
            if ml_type == "Classification" and pd_types.is_object_dtype(y.dtype):
                st.info(f"✨ Auto-encoding categorical target column '{target_col}' using Label Encoding.")
                le_target = LabelEncoder()
                y = pd.Series(le_target.fit_transform(y), name=target_col, index=y.index)
                st.session_state.target_encoder = le_target # Save encoder for potential inverse transform later
                st.session_state.target_classes = le_target.classes_ # Save classes for confusion matrix labels

            st.success(f"✅ ML Task Detected: **{ml_type}**")
            
            st.write("**Target Column:**", target_col)
            st.write("**X Shape (Features):**", X.shape)
            st.write("**y Shape (Label):**", y.shape)

            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["ml_type"] = ml_type
            st.session_state["target_col"] = target_col

            if st.button("Next Phase: Train-Test Split ➡️"):
                st.session_state.current_phase = 5
                st.rerun()
        except KeyError:
            st.error(f"Error: Target column '{target_col}' not found in the DataFrame. This might happen if it was dropped in a previous step.")
    else:
        st.warning("Please select a target column to proceed.")
    
# --- PHASE 6: TRAIN-VALIDATION-TEST SPLIT ---
if st.session_state.current_phase == 5 and st.session_state.file_uploaded and all(key in st.session_state for key in ["X", "y", "ml_type"]):
    st.markdown("---")
    st.markdown("### 🔀 Step 6: Split Your Data")

    st.container(border=True).markdown("#### Data Splitting Configuration")
    st.markdown("Specify your desired split percentages:")
    

    col1, col2, col3 = st.columns(3)
    with col1:
        train_size = st.slider("Train %", min_value=50, max_value=90, value=70, step=5, key="train_slider")
    with col2:
        val_size = st.slider("Validation %", min_value=0, max_value=30, value=15, step=5, key="val_slider") # Changed min_value to 0
    with col3:
        test_size = 100 - train_size - val_size
        st.metric("Test % (Remaining)", test_size)

    if train_size + val_size + test_size != 100:
        st.error("🚫 Split percentages must add up to 100%. Current sum: " + str(train_size + val_size + test_size))
    elif st.button("Split the Data", key="split_data_button"):
        X = st.session_state["X"]
        y = st.session_state["y"]
        ml_type = st.session_state["ml_type"]

        initial_rows = X.shape[0]
        # Align X and y and drop rows where either has NaN
        aligned_df = pd.concat([X, y.rename('target_col_temp')], axis=1, join='inner')
        valid_indices = aligned_df.notna().all(axis=1)

        X_cleaned = aligned_df.drop(columns=['target_col_temp'])[valid_indices]
        y_cleaned = aligned_df['target_col_temp'][valid_indices]

        dropped = initial_rows - X_cleaned.shape[0]
        if dropped > 0:
            st.warning(f"⚠️ {dropped} rows dropped due to missing values in features or target. (Original rows: {initial_rows}, Cleaned rows: {X_cleaned.shape[0]})")
        else:
            st.success("✅ No rows dropped due to missing values in features or target.")
        
        if X_cleaned.empty:
            st.error("🚫 After cleaning, no valid data remains for splitting. Please check your data or imputation steps.")
        elif X_cleaned.shape[0] < 2: # Need at least 2 samples to split
            st.error("🚫 Not enough samples to perform a split (minimum 2 samples required).")
        else:
            current_test_size = test_size / 100.0
            current_val_size = val_size / 100.0

            X_temp, y_temp = X_cleaned, y_cleaned
            X_test, y_test = pd.DataFrame(), pd.Series(dtype=y_cleaned.dtype) # Initialize with correct dtype

            # Stratify only if classification and enough classes
            stratify_test = y_temp if ml_type == "Classification" and y_temp.nunique() > 1 else None

            if current_test_size > 0 and X_cleaned.shape[0] * (1 - current_test_size) >= 1 and X_cleaned.shape[0] * current_test_size >= 1: # Ensure at least 1 sample in each split
                try:
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X_cleaned, y_cleaned,
                        test_size=current_test_size,
                        random_state=42,
                        stratify=stratify_test
                    )
                except ValueError as e:
                    st.error(f"❌ Error during test split: {e}. This can happen with very few samples or single-class target and stratification.")
                    st.info("Attempting split without stratification for test set.")
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X_cleaned, y_cleaned,
                        test_size=current_test_size,
                        random_state=42
                    )
            elif current_test_size > 0:
                st.warning("⚠️ Not enough samples to create a meaningful test set. Test set will be empty or very small.")
            
            X_train, y_train = X_temp, y_temp
            X_val, y_val = pd.DataFrame(), pd.Series(dtype=y_cleaned.dtype) # Initialize with correct dtype

            # Stratify only if classification and enough classes for validation split
            stratify_val = y_train if ml_type == "Classification" and y_train.nunique() > 1 else None

            val_total_ratio = current_val_size / (1 - current_test_size) if (1 - current_test_size) > 0 else 0
            
            if val_total_ratio > 0 and X_temp.shape[0] * (1 - val_total_ratio) >= 1 and X_temp.shape[0] * val_total_ratio >= 1: # Ensure at least 1 sample in each split
                try:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp,
                        test_size=val_total_ratio,
                        random_state=42,
                        stratify=stratify_val
                    )
                except ValueError as e:
                    st.error(f"❌ Error during validation split: {e}. This can happen with very few samples or single-class target and stratification.")
                    st.info("Attempting split without stratification for validation set.")
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp,
                        test_size=val_total_ratio,
                        random_state=42
                    )
            elif current_val_size > 0:
                st.warning("⚠️ Not enough samples to create a meaningful validation set. Validation set will be empty or very small.")

            st.session_state["X_train"] = X_train
            st.session_state["X_val"] = X_val
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_val"] = y_val
            st.session_state["y_test"] = y_test

            st.success("✅ Data has been successfully split!")
            st.write(f"**Train Set:** {X_train.shape[0]} rows ({train_size}%)")
            st.write(f"**Validation Set:** {X_val.shape[0]} rows ({val_size}%)")
            st.write(f"**Test Set:** {X_test.shape[0]} rows ({test_size}%)")
            
            st.button(
                "Next Phase: Model Selection & Training ➡️",
                key="next_to_model_training",
                on_click=next_to_model_training_callback
            )

# --- PHASE 7: MODEL SELECTION & TRAINING ---
if st.session_state.current_phase == 6:
    st.markdown("---")
    st.markdown("### 🧪 Step 7: Model Selection & Training")

    st.write(f"DEBUG (Phase 7): Entered Phase 7. Current phase: {st.session_state.current_phase}")

    # --- Robust Checks for Phase 7 Prerequisites ---
    if not st.session_state.get('file_uploaded', False):
        st.error("Error: No dataset uploaded. Please go back to Phase 1.")
        st.session_state.current_phase = 0
        st.rerun()

    required_keys = ["X_train", "y_train", "ml_type"]
    if not all(key in st.session_state for key in required_keys):
        missing_keys = [key for key in required_keys if key not in st.session_state]
        st.error(f"Error: Missing essential data from previous phases ({', '.join(missing_keys)}). Please ensure data is processed and split correctly in Phase 5 and 6.")
        st.session_state.current_phase = 5
        st.rerun()

    X_train = st.session_state["X_train"]
    y_train = st.session_state["y_train"]
    ml_type = st.session_state["ml_type"]

    if X_train.empty or y_train.empty:
        st.error("Error: Training data (X_train or y_train) is empty after splitting. This can happen if your dataset is too small or if too many rows were dropped. Please review Phase 3 and 6.")
        st.session_state.current_phase = 5
        st.rerun()
    # --- End of Robust Checks ---


    st.write(f"DEBUG (Phase 7): All prerequisites met. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, ML Type: **{ml_type}**")

    # --- Model Definitions ---
    classification_models_dict = {
        "Logistic Regression": {"model": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42), "desc": "Simple, interpretable model for binary classification. Good baseline."},
        "Decision Tree Classifier": {"model": DecisionTreeClassifier(random_state=42), "desc": "Tree-like model, easy to interpret, handles non-linear relationships."},
        "Random Forest Classifier": {"model": RandomForestClassifier(random_state=42), "desc": "Ensemble of decision trees, generally more accurate and stable, reduces overfitting."},
        "Support Vector Classifier (SVC)": {"model": SVC(probability=True, random_state=42, max_iter=10000), "desc": "Effective in high-dimensional spaces, finds the best boundary to separate classes."},
        "K-Nearest Neighbors (KNN)": {"model": KNeighborsClassifier(), "desc": "Classifies based on the majority class of its nearest neighbors. Simple, non-parametric."},
        "Naive Bayes (GaussianNB)": {"model": GaussianNB(), "desc": "Based on Bayes' theorem, good for text classification and high-dimensional data, assumes feature independence."},
        "Gradient Boosting (XGBoost Classifier)": {"model": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), "desc": "Advanced boosting algorithm, highly performant, often wins competitions."},
        "Gradient Boosting (LightGBM Classifier)": {"model": lgb.LGBMClassifier(random_state=42), "desc": "Fast and efficient gradient boosting, good for large datasets."},
        "Artificial Neural Network (MLP Classifier)": {"model": MLPClassifier(max_iter=1000, random_state=42), "desc": "Foundation of deep learning, good for complex non-linear patterns. Can be computationally intensive."}
    }

    regression_models_dict = {
        "Linear Regression": {"model": LinearRegression(), "desc": "Simple, interpretable model for predicting continuous values. Good baseline."},
        "Decision Tree Regressor": {"model": DecisionTreeRegressor(random_state=42), "desc": "Tree-like model for regression, easy to interpret, handles non-linear relationships."},
        "Random Forest Regressor": {"model": RandomForestRegressor(random_state=42), "desc": "Ensemble of decision trees, generally more accurate and stable, reduces overfitting."},
        "Support Vector Regressor (SVR)": {"model": SVR(max_iter=10000), "desc": "Effective in high-dimensional spaces, finds the best boundary to fit data points."},
        "K-Nearest Neighbors (KNN)": {"model": KNeighborsRegressor(), "desc": "Predicts based on the average value of its nearest neighbors. Simple, non-parametric."},
        "Gradient Boosting (XGBoost Regressor)": {"model": xgb.XGBRegressor(random_state=42), "desc": "Advanced boosting algorithm, highly performant, often wins competitions."},
        "Gradient Boosting (LightGBM Regressor)": {"model": lgb.LGBMRegressor(random_state=42), "desc": "Fast and efficient gradient boosting, good for large datasets."},
        "Artificial Neural Network (MLP Regressor)": {"model": MLPRegressor(max_iter=1000, random_state=42), "desc": "Foundation of deep learning, good for complex non-linear patterns. Can be computationally intensive."}
    }

    st.container(border=True).markdown("#### 🤖 Choose Machine Learning Models")
    selected_model_names = []
    if ml_type == "Classification":
        st.subheader("Classification Models:")
        selected_model_names = st.multiselect(
            "Select one or more classification models to train:",
            list(classification_models_dict.keys()),
            key="classification_model_select"
        )
    elif ml_type == "Regression":
        st.subheader("Regression Models:")
        selected_model_names = st.multiselect(
            "Select one or more regression models to train:",
            list(regression_models_dict.keys()),
            key="regression_model_select"
        )
    else:
        st.warning("ML task could not be determined. Please go back to Phase 5.")

    if selected_model_names:
        st.markdown("---")
        st.container(border=True).markdown("#### ⚙️ Adjust Model Hyperparameters (Optional)")
        model_hyperparams = {}

        for model_name in selected_model_names:
            st.markdown(f"##### {model_name} Hyperparameters:")
            with st.expander(f"Configure {model_name}"):
                params = {}
                if ml_type == "Classification":
                    model_info = classification_models_dict[model_name]
                    st.info(f"**Recommendation:** {model_info['desc']}")
                    if model_name == "Logistic Regression":
                        params['C'] = st.slider("C (Inverse of regularization strength)", 0.01, 10.0, 1.0, 0.01, key=f"lr_c_{model_name}")
                    elif model_name == "Decision Tree Classifier":
                        params['max_depth'] = st.slider("Max Depth", 1, 20, 10, key=f"dtc_md_{model_name}")
                        params['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2, key=f"dtc_mss_{model_name}")
                    elif model_name == "Random Forest Classifier":
                        params['n_estimators'] = st.slider("Number of Estimators", 10, 200, 100, 10, key=f"rfc_ne_{model_name}")
                        params['max_depth'] = st.slider("Max Depth", 1, 20, 10, key=f"rfc_md_{model_name}")
                    elif model_name == "Support Vector Classifier (SVC)":
                        params['C'] = st.slider("C (Regularization parameter)", 0.1, 10.0, 1.0, 0.1, key=f"svc_c_{model_name}")
                        params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'], key=f"svc_k_{model_name}")
                    elif model_name == "K-Nearest Neighbors (KNN)":
                        params['n_neighbors'] = st.slider("Number of Neighbors (k)", 1, 20, 5, key=f"knn_nn_{model_name}")
                    elif model_name == "Naive Bayes (GaussianNB)":
                        st.info("Gaussian Naive Bayes has few hyperparameters for tuning in this basic setup.")
                    elif model_name == "Gradient Boosting (XGBoost Classifier)":
                        params['n_estimators'] = st.slider("Number of Estimators", 50, 500, 100, 50, key=f"xgbc_ne_{model_name}")
                        params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key=f"xgbc_lr_{model_name}")
                        params['max_depth'] = st.slider("Max Depth", 1, 10, 3, key=f"xgbc_md_{model_name}")
                    elif model_name == "Gradient Boosting (LightGBM Classifier)":
                        params['n_estimators'] = st.slider("Number of Estimators", 50, 500, 100, 50, key=f"lgbmc_ne_{model_name}")
                        params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key=f"lgbmc_lr_{model_name}")
                        params['max_depth'] = st.slider("Max Depth", -1, 10, -1, key=f"lgbmc_md_{model_name}") # -1 means no limit
                    elif model_name == "Artificial Neural Network (MLP Classifier)":
                        params['hidden_layer_sizes'] = st.slider("Hidden Layer Size", 10, 200, 100, 10, key=f"mlpc_hls_{model_name}")
                        params['alpha'] = st.slider("Alpha (L2 regularization)", 0.0001, 0.1, 0.0001, 0.0001, key=f"mlpc_alpha_{model_name}", format="%.4f")


                elif ml_type == "Regression":
                    model_info = regression_models_dict[model_name]
                    st.info(f"**Recommendation:** {model_info['desc']}")
                    if model_name == "Linear Regression":
                        st.info("Linear Regression has no common hyperparameters to tune in this basic setup.")
                    elif model_name == "Decision Tree Regressor":
                        params['max_depth'] = st.slider("Max Depth", 1, 20, 10, key=f"dtr_md_{model_name}")
                        params['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2, key=f"dtr_mss_{model_name}")
                    elif model_name == "Random Forest Regressor":
                        params['n_estimators'] = st.slider("Number of Estimators", 10, 200, 100, 10, key=f"rfr_ne_{model_name}")
                        params['max_depth'] = st.slider("Max Depth", 1, 20, 10, key=f"rfr_md_{model_name}")
                    elif model_name == "Support Vector Regressor (SVR)":
                        params['C'] = st.slider("C (Regularization parameter)", 0.1, 10.0, 1.0, 0.1, key=f"svr_c_{model_name}")
                        params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'], key=f"svr_k_{model_name}")
                    elif model_name == "K-Nearest Neighbors (KNN)":
                        params['n_neighbors'] = st.slider("Number of Neighbors (k)", 1, 20, 5, key=f"knnr_nn_{model_name}")
                    elif model_name == "Gradient Boosting (XGBoost Regressor)":
                        params['n_estimators'] = st.slider("Number of Estimators", 50, 500, 100, 50, key=f"xgbr_ne_{model_name}")
                        params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key=f"xgbr_lr_{model_name}")
                        params['max_depth'] = st.slider("Max Depth", 1, 10, 3, key=f"xgbr_md_{model_name}")
                    elif model_name == "Gradient Boosting (LightGBM Regressor)":
                        params['n_estimators'] = st.slider("Number of Estimators", 50, 500, 100, 50, key=f"lgbmr_ne_{model_name}")
                        params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key=f"lgbmr_lr_{model_name}")
                        params['max_depth'] = st.slider("Max Depth", -1, 10, -1, key=f"lgbmr_md_{model_name}") # -1 means no limit
                    elif model_name == "Artificial Neural Network (MLP Regressor)":
                        params['hidden_layer_sizes'] = st.slider("Hidden Layer Size", 10, 200, 100, 10, key=f"mlpr_hls_{model_name}")
                        params['alpha'] = st.slider("Alpha (L2 regularization)", 0.0001, 0.1, 0.0001, 0.0001, key=f"mlpr_alpha_{model_name}", format="%.4f")
                
                model_hyperparams[model_name] = params

        st.markdown("---")
        if st.button("Train Selected Models 🚀", key="train_models_button"):
            st.session_state.trained_models = {}
            
            if X_train.empty or y_train.empty:
                st.error("Training data (X_train or y_train) is empty. Please go back to previous phases to ensure data is correctly processed and split.")
            else:
                progress_text = "Training models and generating learning curves. Please wait..."
                my_bar = st.progress(0, text=progress_text)
                
                for i, model_name in enumerate(selected_model_names):
                    my_bar.progress((i + 1) / len(selected_model_names), text=f"Training {model_name}...")
                    
                    try:
                        model = None
                        if ml_type == "Classification":
                            model = classification_models_dict[model_name]["model"]
                            scoring_metric = 'accuracy'
                        elif ml_type == "Regression":
                            model = regression_models_dict[model_name]["model"]
                            scoring_metric = 'r2'

                        # Set hyperparameters if they were adjusted
                        if model_name in model_hyperparams:
                            model.set_params(**model_hyperparams[model_name])

                        start_time = time.time()
                        model.fit(X_train, y_train)
                        end_time = time.time()
                        training_duration = end_time - start_time

                        st.success(f"✅ {model_name} trained successfully in {training_duration:.2f} seconds!")

                        # --- Generate Learning Curve Data ---
                        with st.spinner(f"Generating learning curve for {model_name}..."):
                            # Adjust train_sizes to avoid issues with very small datasets
                            train_sizes, train_scores, test_scores = learning_curve(
                                model, X_train, y_train, cv=3, n_jobs=-1,
                                train_sizes=np.linspace(0.1, 1.0, 5), # 5 points from 10% to 100% of training data
                                scoring=scoring_metric
                            )
                        st.success(f"✅ Learning curve data generated for {model_name}.")

                        st.session_state.trained_models[model_name] = {
                            "model": model,
                            "training_duration": training_duration,
                            "ml_type": ml_type,
                            "learning_curve_data": {
                                "train_sizes": train_sizes,
                                "train_scores": train_scores,
                                "test_scores": test_scores,
                                "scoring_metric": scoring_metric
                            }
                        }

                    except Exception as e:
                        st.error(f"❌ Error training or generating learning curve for {model_name}: {e}")
                
                my_bar.empty()
                st.success("All selected models have been trained and learning curves generated!")
                
                st.button(
                    "Next Phase: Model Evaluation & Comparison 📈",
                    key="next_to_model_evaluation",
                    on_click=next_to_model_evaluation_callback
                )
    else:
        st.info("Please select at least one model to train.")
    
# --- PHASE 8: MODEL EVALUATION & COMPARISON ---
if st.session_state.current_phase == 7: # Simplified condition for debugging
    st.markdown("---")
    st.markdown("### 📈 Step 8: Model Evaluation & Comparison")

    st.write(f"DEBUG (Phase 8): Entered Phase 8. Current phase: {st.session_state.current_phase}")

    # --- Robust Checks for Phase 8 Prerequisites ---
    if not st.session_state.get('file_uploaded', False):
        st.error("Error: No dataset uploaded. Please go back to Phase 1.")
        st.session_state.current_phase = 0
        st.rerun()

    required_keys_p8 = ["X_test", "y_test", "ml_type", "trained_models"]
    if not all(key in st.session_state for key in required_keys_p8):
        missing_keys = [key for key in required_keys_p8 if key not in st.session_state]
        st.error(f"Error: Missing essential data from previous phases ({', '.join(missing_keys)}). Please ensure data is processed, split, and models are trained correctly.")
        st.session_state.current_phase = 6 # Redirect to model training phase
        st.rerun()

    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    ml_type = st.session_state["ml_type"]
    trained_models = st.session_state["trained_models"]
    target_classes = st.session_state.get('target_classes', None) # Get original class labels if target was encoded

    if X_test.empty or y_test.empty:
        st.warning("Test data is empty. Cannot perform model evaluation. Please ensure your data split is valid in Phase 6.")
        st.session_state.current_phase = 6 # Redirect to split phase
        st.rerun()
    elif not trained_models:
        st.warning("No models have been trained yet. Please go back to Phase 7 to train models.")
        st.session_state.current_phase = 6 # Redirect to model training phase
        st.rerun()
    # --- End of Robust Checks ---

    st.container(border=True).markdown("#### Model Performance Summary")
    metrics_data = []

    for model_name, model_info in trained_models.items():
        model = model_info["model"]
        
        try:
            y_pred = model.predict(X_test)
            
            row_metrics = {"Model": model_name, "Training Duration (s)": f"{model_info['training_duration']:.2f}"}

            if ml_type == "Classification":
                if len(np.unique(y_test)) < 2:
                    st.warning(f"Skipping detailed classification metrics for {model_name}: Test set has only one class.")
                    row_metrics.update({
                        "Accuracy": "N/A", "Precision": "N/A", "Recall": "N/A", "F1-Score": "N/A"
                    })
                else:
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    row_metrics.update({
                        "Accuracy": f"{accuracy:.4f}",
                        "Precision": f"{precision:.4f}",
                        "Recall": f"{recall:.4f}",
                        "F1-Score": f"{f1:.4f}"
                    })
                    
            elif ml_type == "Regression":
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                row_metrics.update({
                    "MAE": f"{mae:.4f}",
                    "MSE": f"{mse:.4f}",
                    "R² Score": f"{r2:.4f}"
                })
            metrics_data.append(row_metrics)
        except Exception as e:
            st.error(f"Error evaluating {model_name}: {e}")
            metrics_data.append({"Model": model_name, "Error": str(e)})

    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)

        # Sort the DataFrame based on ML Type performance metric
        if ml_type == "Classification":
            # Convert 'Accuracy' to numeric, putting 'N/A' at the end
            metrics_df['Accuracy_sort'] = pd.to_numeric(metrics_df['Accuracy'], errors='coerce')
            metrics_df = metrics_df.sort_values(by='Accuracy_sort', ascending=False, na_position='last')
            metrics_df = metrics_df.drop(columns='Accuracy_sort') # Drop helper column
        elif ml_type == "Regression":
            # Convert 'R² Score' to numeric, putting 'N/A' at the end
            metrics_df['R2_sort'] = pd.to_numeric(metrics_df['R² Score'], errors='coerce')
            metrics_df = metrics_df.sort_values(by='R2_sort', ascending=False, na_position='last')
            metrics_df = metrics_df.drop(columns='R2_sort') # Drop helper column
            
        st.dataframe(metrics_df, use_container_width=True)
    
    st.markdown("---")
    st.container(border=True).markdown("#### Detailed Model Visualizations")
    selected_model_for_viz = st.selectbox(
        "Select a model for detailed visualization:",
        list(st.session_state.trained_models.keys()),
        key="viz_model_select"
    )

    if selected_model_for_viz:
        model_info = st.session_state.trained_models[selected_model_for_viz]
        model = model_info["model"]
        ml_type_viz = model_info["ml_type"] # Use ml_type from stored model info

        if ml_type_viz == "Classification":
            st.markdown(f"##### {selected_model_for_viz} - Confusion Matrix")
            if len(np.unique(y_test)) < 2:
                st.info("Cannot plot Confusion Matrix: Test set has only one class.")
            else:
                try:
                    cm = confusion_matrix(y_test, model.predict(X_test))
                    # Use original class labels for better interpretability if available
                    cm_labels = target_classes if target_classes is not None else np.unique(y_test)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                 xticklabels=cm_labels, yticklabels=cm_labels)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error plotting confusion matrix: {e}")

            st.markdown(f"##### {selected_model_for_viz} - ROC Curve")
            if len(np.unique(y_test)) < 2:
                st.info("Cannot plot ROC Curve: Test set has only one class.")
            elif len(np.unique(y_test)) > 2:
                st.info("ROC Curve is typically for binary classification. Skipping for multi-class.")
            else:
                try:
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)

                        fig, ax = plt.subplots(figsize=(6, 5))
                        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                        ax.legend(loc="lower right")
                        st.pyplot(fig)
                    else:
                        st.info(f"Model {selected_model_for_viz} does not support `predict_proba` for ROC curve.")
                except Exception as e:
                    st.error(f"Error plotting ROC curve: {e}.")

        elif ml_type_viz == "Regression":
            st.markdown(f"##### {selected_model_for_viz} - Residual Plot")
            try:
                y_pred = model.predict(X_test)
                residuals = y_test - y_pred
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=y_pred, y=residuals, ax=ax)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                ax.set_title('Residual Plot')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting residual plot: {e}")

            st.markdown(f"##### {selected_model_for_viz} - Actual vs. Predicted Scatter Plot")
            try:
                y_pred = model.predict(X_test)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=y_test, y=y_pred, ax=ax)
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title('Actual vs. Predicted Values')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting actual vs. predicted plot: {e}")
        
        # --- Learning Curve Plotting ---
        if "learning_curve_data" in model_info and model_info["learning_curve_data"]:
            st.markdown(f"##### {selected_model_for_viz} - Learning Curve")
            lc_data = model_info["learning_curve_data"]
            train_sizes = lc_data["train_sizes"]
            train_scores = lc_data["train_scores"]
            test_scores = lc_data["test_scores"]
            scoring_metric_name = lc_data["scoring_metric"]

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            fig_lc, ax_lc = plt.subplots(figsize=(8, 6))
            ax_lc.fill_between(train_sizes, train_scores_mean - train_scores_std,
                               train_scores_mean + train_scores_std, alpha=0.1,
                               color="r")
            ax_lc.fill_between(train_sizes, test_scores_mean - test_scores_std,
                               test_scores_mean + test_scores_std, alpha=0.1,
                               color="g")
            ax_lc.plot(train_sizes, train_scores_mean, 'o-', color="r",
                       label="Training score")
            ax_lc.plot(train_sizes, test_scores_mean, 'o-', color="g",
                       label="Cross-validation score")
            ax_lc.set_xlabel("Training examples")
            ax_lc.set_ylabel(f"Score ({scoring_metric_name})")
            ax_lc.set_title(f"Learning Curve for {selected_model_for_viz}")
            ax_lc.legend(loc="best")
            st.pyplot(fig_lc)
        else:
            st.info(f"Learning curve data not available for {selected_model_for_viz}.")


    st.markdown("---")
    if st.button("Next Phase: Model Saving 💾", key="next_to_model_saving"):
        st.session_state.current_phase = 8
        st.rerun()

# --- PHASE 9: MODEL SAVING ---
if st.session_state.current_phase == 8 and st.session_state.file_uploaded and st.session_state.trained_models:
    st.markdown("---")
    st.markdown("### 💾 Step 9: Save Your Trained Model")

    if not st.session_state.trained_models:
        st.warning("No models have been trained yet. Please go back to Phase 7 to train models.")
    else:
        st.container(border=True).markdown("#### Select Model to Save")
        model_to_save_name = st.selectbox(
            "Choose which trained model you want to save:",
            list(st.session_state.trained_models.keys()),
            key="save_model_select"
        )

        if model_to_save_name:
            model_info = st.session_state.trained_models[model_to_save_name]
            model = model_info["model"]

            # Generate a unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_model_name = model_to_save_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") # Added replace for hyphen
            file_name = f"{safe_model_name}_{timestamp}.joblib"
            file_path = os.path.join("models", file_name)

            st.write(f"You've selected **{model_to_save_name}** to save.")
            st.info(f"The model will be saved as `{file_name}` in the `models/` directory.")

            if st.button(f"Save and Download {model_to_save_name}", key="download_model_button"):
                try:
                    joblib.dump(model, file_path)
                    st.success(f"✅ Model '{model_to_save_name}' saved successfully to `{file_path}`!")

                    # Provide download button
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label=f"Download {file_name}",
                            data=f,
                            file_name=file_name,
                            mime="application/octet-stream",
                            key="final_download_button"
                        )
                    st.info("You can now download the model file to your local machine.")

                except Exception as e:
                    st.error(f"❌ Error saving the model: {e}")
        else:
            st.info("Please select a model to enable saving.")
    
    st.markdown("---")
    if st.button("Start Over with a New Dataset 🔄", key="start_over_button_phase9"):
        st.session_state.clear() # Clear all session state variables
        st.rerun()

# --- PHASE 10: RESET / START OVER ---
# This phase is primarily handled by the "Start Over" button which can be placed at the end of the pipeline.
# For simplicity, I've integrated a "Start Over" button at the end of Phase 9.