# sqft = [1700.0, 1900.0, 2100.0, 2900.0, 2700.0]
# bedrooms = [3.0, 3.0, 3.0, 4.0, 4.0]
# location_score = [6.0, 8.0, 10.0, 11.0, 9.0]
# price = [250000.0, 280000.0, 310000.0, 400000.0, 360000.0]


sqft = [5.0, 10.0, 15.0, 20.0, 25.0]
bedrooms = [3.0, 3.0, 3.0, 4.0, 4.0]
location_score = [6.0, 8.0, 10.0, 11.0, 9.0]
price = [34, 45, 56, 66, 65]


slope_sqft = 1.0
slope_bedrooms = 1.0
slope_location_score = 1.0
intercept = 0.0
learning_rate = 0.0001


# the function is supposed to take floats as inputs, not lists
def predict_price(float_sqft, float_bedrooms, float_location_score):
    return float(float_sqft * slope_sqft + float_bedrooms * slope_bedrooms +
                 float_location_score * slope_location_score + intercept)


# the function is supposed to take lists as inputs, not floats
def sum_of_squared_residuals():
    loss = 0.0
    for i in range(len(price)):
        loss += (price[i] - predict_price(sqft[i], bedrooms[i], location_score[i])) ** 2
    return loss


def sum_of_predict_derivatives_w_respect_to_slope_sqft():
    result = 0.0
    for i in range(len(sqft)):
        result += 2*(price[i] - predict_price(sqft[i], bedrooms[i], location_score[i]))*sqft[i]
    return result


def sum_of_predict_derivatives_w_respect_to_slope_bedrooms():
    result = 0.0
    for i in range(len(sqft)):
        result += 2*(price[i] - predict_price(sqft[i], bedrooms[i], location_score[i]))*bedrooms[i]
    return result


def sum_of_predict_derivatives_w_respect_to_slope_location_score():
    result = 0.0
    for i in range(len(sqft)):
        result += 2*(price[i] - predict_price(sqft[i], bedrooms[i], location_score[i]))*location_score[i]
    return result


def sum_of_predict_derivatives_w_respect_to_intercept():
    result = 0.0
    for i in range(len(sqft)):
        result += 2*(price[i] - predict_price(sqft[i], bedrooms[i], location_score[i]))
    return result


def train():
    global slope_sqft, slope_bedrooms, slope_location_score, intercept
    while sum_of_squared_residuals() > 1e-6:
        slope_sqft += sum_of_predict_derivatives_w_respect_to_slope_sqft() * learning_rate
        slope_bedrooms += sum_of_predict_derivatives_w_respect_to_slope_bedrooms() * learning_rate
        slope_location_score += sum_of_predict_derivatives_w_respect_to_slope_location_score() * learning_rate
        intercept += sum_of_predict_derivatives_w_respect_to_intercept() * learning_rate


train()
print(slope_sqft)
print(slope_bedrooms)
print(slope_location_score)
print(intercept)
print(predict_price(23, 7, 10))
