"""
=====================================================
 Multi-Variable Linear Regression from Scratch
=====================================================

This script demonstrates a basic implementation of gradient descent
for a multi-variable linear regression problem using only Python and
no external machine learning libraries.

It includes:
 - Input features: square footage, number of bedrooms, location score
 - Target values: house prices
 - Prediction function for a linear model with multiple features
 - Calculation of sum of squared residuals (loss)
 - Derivatives of loss with respect to each slope and intercept
 - Gradient descent loop to optimize parameters

Author: Dennis Alacahanli
Purpose: An early linear regression model I implemented to understand training with multiple variables
"""

# ----------------- Dataset ----------------- #
sqft = [5.0, 10.0, 15.0, 20.0, 25.0]          # Feature 1: square footage
bedrooms = [3.0, 3.0, 3.0, 4.0, 4.0]          # Feature 2: number of bedrooms
location_score = [6.0, 8.0, 10.0, 11.0, 9.0]  # Feature 3: location score
price = [340000, 450000, 560000, 660000, 650000]                  # Target: house prices

# ----------------- Model Parameters ----------------- #
slope_sqft = 1.0            # Initial slope for sqft
slope_bedrooms = 1.0        # Initial slope for bedrooms
slope_location_score = 1.0  # Initial slope for location score
intercept = 0.0             # Initial intercept
learning_rate = 0.001      # Learning rate for gradient descent

# ----------------- Prediction Function ----------------- #
def predict_price(float_sqft, float_bedrooms, float_location_score):
    """
    Compute predicted price using current slopes and intercept.
    """
    return float(float_sqft * slope_sqft + float_bedrooms * slope_bedrooms +
                 float_location_score * slope_location_score + intercept)

# ----------------- Loss Function ----------------- #
def sum_of_squared_residuals():
    """
    Compute the sum of squared differences between predicted and actual prices.
    """
    loss = 0.0
    for i in range(len(price)):
        loss += (price[i] - predict_price(sqft[i], bedrooms[i], location_score[i])) ** 2
    return loss

# ----------------- Gradients ----------------- #
def sum_of_predict_derivatives_w_respect_to_slope_sqft():
    """
    Derivative of loss with respect to slope for sqft.
    """
    result = 0.0
    for i in range(len(sqft)):
        result += 2*(price[i] - predict_price(sqft[i], bedrooms[i], location_score[i]))*sqft[i]
    return result

def sum_of_predict_derivatives_w_respect_to_slope_bedrooms():
    """
    Derivative of loss with respect to slope for bedrooms.
    """
    result = 0.0
    for i in range(len(sqft)):
        result += 2*(price[i] - predict_price(sqft[i], bedrooms[i], location_score[i]))*bedrooms[i]
    return result

def sum_of_predict_derivatives_w_respect_to_slope_location_score():
    """
    Derivative of loss with respect to slope for location score.
    """
    result = 0.0
    for i in range(len(sqft)):
        result += 2*(price[i] - predict_price(sqft[i], bedrooms[i], location_score[i]))*location_score[i]
    return result

def sum_of_predict_derivatives_w_respect_to_intercept():
    """
    Derivative of loss with respect to intercept.
    """
    result = 0.0
    for i in range(len(sqft)):
        result += 2*(price[i] - predict_price(sqft[i], bedrooms[i], location_score[i]))
    return result

# ----------------- Gradient Descent ----------------- #
def train():
    """
    Perform gradient descent to optimize slopes and intercept.
    """
    global slope_sqft, slope_bedrooms, slope_location_score, intercept
    print("training...")
    while sum_of_squared_residuals() > 1e-6:
        slope_sqft += sum_of_predict_derivatives_w_respect_to_slope_sqft() * learning_rate
        slope_bedrooms += sum_of_predict_derivatives_w_respect_to_slope_bedrooms() * learning_rate
        slope_location_score += sum_of_predict_derivatives_w_respect_to_slope_location_score() * learning_rate
        intercept += sum_of_predict_derivatives_w_respect_to_intercept() * learning_rate

# ----------------- Train the Model ----------------- #
train()

# ----------------- Predict Example ----------------- #
print(f'this house should cost about: {predict_price(5, 3, 6)}$')  # Predict price for a new house
