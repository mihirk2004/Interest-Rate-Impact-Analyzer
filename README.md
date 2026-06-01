# Interest Rate and Economic Indicators Prediction

This project is a multi-page Streamlit application for exploring how interest rates relate to major economic indicators such as GDP, inflation, loan rates, savings, and economic growth. It combines notebook-based model training, saved model artifacts, and a dashboard-style UI so users can interact with each prediction task from one home screen.

The project is centered on the idea that interest rates influence several dependent financial variables, and each notebook focuses on one relationship with a dedicated regression model and visual analysis.

## What This Project Does

- Predicts economic variables from user-selected economic inputs.
- Presents a Streamlit home dashboard with 8 topic cards.
- Uses different regression approaches depending on the target relationship.
- Includes visual explanations such as actual-vs-predicted plots, trend lines, feature-importance charts, and relationship curves.
- Separates training work into notebook files under `dependencies/` and deployment logic into `index.py` and `pages/`.

## Prediction Modules

| Relationship | Model Used | Main Output Style |
| --- | --- | --- |
| GDP Per Capita -> Economic Growth | GradientBoostingRegressor | Prediction dashboard plus historical GDP vs growth plot |
| Inflation Rate -> Economic Growth | XGBoost Regressor | Slider-based prediction plus historical trend visualization |
| Inflation Rate -> Loan Rate | XGBoost Regressor | Prediction panel with feature generation and time-series chart |
| Interest Rate -> Economic Growth | LSTM neural network | Sequence-based prediction with historical interest-rate and growth plot |
| Interest Rate -> GDP Per Capita | LSTM neural network | Sequence prediction with historical trend comparison |
| Interest Rate -> Inflation Rate | XGBoost Regressor | Input-driven prediction with relationship curve |
| Interest Rate -> Loan Rate | Neural network with polynomial features | Curved prediction plot and feature-importance-style weights |
| Interest Rate -> Savings Rate | XGBoost Regressor | Multi-input prediction with savings relationship curve |

## Model And Notebook Summary

The notebooks in `dependencies/` are the training and experimentation layer of the project.

- `dependencies/gdp and economic growth/gdp and economic growth.ipynb` trains a `GradientBoostingRegressor` and saves it as `economic_growth_model.pkl`.
- `dependencies/inflation and economic growth/inflation and economic growth.ipynb` trains an XGBoost model for economic growth prediction from inflation features and saves `economic_growth_xgboost_model.pkl`.
- `dependencies/inflation and loan/inflation and loan.ipynb` trains an XGBoost regressor for loan-rate prediction from inflation-based lag features and saves `inflation_loan_xgboost_model.pkl`.
- `dependencies/interest rate and gdp/interest and gdp.ipynb` trains an LSTM model for GDP per capita prediction and saves `lstm_gdp_predictor.h5`.
- `dependencies/interest rate and economic growth/interest rate and economic growth.ipynb` trains an enhanced LSTM sequence model for interest-rate to growth prediction and saves `improved_economic_growth_lstm.keras`.
- `dependencies/interest rate and inflation/interest rate and inflation.ipynb` trains an XGBoost regressor for inflation prediction from interest-rate features and saves `optimized_inflation_model.pkl`.
- `dependencies/interest rate and loan/interest rate and loan rate.ipynb` contains a dense neural-network workflow for loan-rate prediction and uses scaled / engineered features.
- `dependencies/interest rate and savings/interest rate and savings.ipynb` trains an XGBoost regressor for savings-rate prediction from interest-rate and date features.

## Streamlit App Structure

The main entry point is `index.py`. It builds the landing page, applies custom styling, and routes users to the prediction pages.

The home page shows 8 cards:

1. GDP and Economic Growth
2. Inflation Rate and Economic Growth
3. Inflation Rate and Loan Rate
4. Interest Rate and Economic Growth
5. Interest Rate and GDP
6. Interest Rate and Inflation Rate
7. Interest Rate and Loan Rate
8. Interest Rate and Saving Rate

The app uses `streamlit_option_menu` for the top navigation bar and `st.switch_page()` to move between pages in the `pages/` folder.

## Visual Design

The project includes custom styling through `style.css` and image handling via `Pillow` and base64 encoding. The dashboard layout uses card-based navigation, wide layout settings, and simple footer sections to keep the app readable and presentation-friendly.

Across the notebooks and dashboards, the plots are designed to be practical rather than decorative:

- Actual vs predicted comparisons for model quality checks.
- Time-series trend charts for historical context.
- Feature-importance plots for tree-based models.
- Relationship curves that show how the predicted output changes as a single input changes.
- Sequence-model plots for LSTM workflows.

## Tech Stack

- Python
- Streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- TensorFlow / Keras
- matplotlib
- seaborn
- joblib
- Pillow

## Installation

1. Clone the repository.
2. Create and activate a Python environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run index.py
```

## Important Note About Paths

Several page files and notebooks currently use absolute Windows paths such as `E:\Admin\User\...`. If you clone the project to a different folder or machine, update those paths to match your local setup or convert them to relative paths before running the app.

## Project Structure

```text
Interest-Rate-Impact-Analyzer/
|-- index.py
|-- pages/
|-- dependencies/
|   |-- notebooks, CSV files, and model artifacts
|-- datasets/
|-- models/
|-- images/
|-- style.css
|-- style1.css
|-- requirements.txt
```
## Frontend 

![System Diagram](Group%206/Screenshot%20(124).png)
![System Diagram](Group%206/Screenshot%20(126).png)
![System Diagram](Group%206/Screenshot%20(136).png)
![System Diagram](Group%206/Screenshot%20(139).png)

## Future Improvements

- Replace absolute file paths with relative paths for easier reuse.
- Add evaluation metrics directly on each Streamlit page.
- Standardize model naming and artifact locations.
- Add screenshots or exported plots for the README gallery section.
- Add a single config layer for dataset and model paths.

## Acknowledgement

This project was built as an academic assignment focused on predicting financial and macroeconomic indicators using machine learning.
