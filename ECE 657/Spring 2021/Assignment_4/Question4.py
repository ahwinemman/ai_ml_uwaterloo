import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt

# Problem: define angle and distance (inputs) and speed and steering (outputs)
x_angle = np.arange(0, 91, 1)
x_distance = np.arange(0, 10.5, 0.5)
x_speed = np.arange(0, 5.2, 0.2)
x_steering = np.arange(0, 91, 1)

# Membership functions for angle
angle_small = fuzz.trimf(x_angle, [0, 0, 60])
angle_medium = fuzz.trimf(x_angle, [0, 60, 90])
angle_large = fuzz.trimf(x_angle, [60, 90, 90])

# for distance
distance_near = fuzz.trimf(x_distance, [0, 0, 5])
distance_far = fuzz.trimf(x_distance, [0, 5, 10])
distance_veryfar = fuzz.trimf(x_distance, [5, 10, 10])

# for steering
steering_medium = fuzz.trimf(x_steering, [0, 0, 60])
steering_sharp = fuzz.trimf(x_steering, [0, 60, 90])
steering_verysharp = fuzz.trimf(x_steering, [60, 90, 90])

# for speed
speed_low = fuzz.trimf(x_speed, [0, 1, 2])
speed_medium = fuzz.trimf(x_speed, [1, 2, 3])
speed_fast = fuzz.trimf(x_speed, [2, 3, 4])
speed_maximum = fuzz.trimf(x_speed, [3, 4, 5])

# Input: define angle and distance
angle = 20
distance = 2

angle_small_degree = fuzz.interp_membership(
    x_angle, angle_small, angle)
angle_medium_degree = fuzz.interp_membership(
    x_angle, angle_medium, angle)
angle_large_degree = fuzz.interp_membership(
    x_angle, angle_large, angle)

distance_near_degree = fuzz.interp_membership(
    x_distance, distance_near, distance)
distance_far_degree = fuzz.interp_membership(
    x_distance, distance_far, distance)
distance_veryfar_degree = fuzz.interp_membership(
    x_distance, distance_veryfar, distance)

fig_scale_x = 2.0
fig_scale_y = 1.5
fig = plt.figure(figsize=(6.4 * fig_scale_x, 4.8 * fig_scale_y))
row = 2
col = 3

# plot the membership functions
plt.subplot(row, col, 1)
plt.title("Angle")
plt.plot(x_angle, angle_small, label="small", marker=".")
plt.plot(x_angle, angle_medium, label="medium", marker=".")
plt.plot(x_angle, angle_large, label="large", marker=".")
plt.plot(angle, 0.0, label="angle", marker="D")
plt.legend(loc="upper left")

plt.subplot(row, col, 2)
plt.title("Distance")
plt.plot(x_distance, distance_near, label="near", marker=".")
plt.plot(x_distance, distance_far, label="far", marker=".")
plt.plot(x_distance, distance_veryfar, label="very far", marker=".")
plt.plot(distance, 0.0, label="distance", marker="D")
plt.legend(loc="upper left")

plt.subplot(row, col, 3)
plt.title("Speed")
plt.plot(x_speed, speed_low, label="slow", marker=".")
plt.plot(x_speed, speed_medium, label="medium", marker=".")
plt.plot(x_speed, speed_fast, label="fast", marker=".")
plt.plot(x_speed, speed_maximum, label="max", marker=".")
plt.legend(loc="upper left")

plt.subplot(row, col, 4)
plt.title("Steering")
plt.plot(x_steering, steering_medium, label="mild", marker=".")
plt.plot(x_steering, steering_sharp, label="sharp", marker=".")
plt.plot(x_steering, steering_verysharp, label="very sharp", marker=".")
plt.legend(loc="upper left")

# ==================== FOR STEERING ===================================
# use min since we have AND for the rule base.
# Comment out the Speed section when graphing
# for steering since the graph can hold a max of 6 plots. Like what is done right now.

# avoid obstacles when near
steering_vsharpdegree = np.fmin(angle_small_degree, distance_near_degree)
# avoid obstacles when medium distance
steering_sharpdegree = np.fmin(angle_medium_degree, distance_far_degree)
# avoid obstacles when far
steering_milddegree = np.fmin(angle_small_degree, distance_veryfar_degree)

activation_high = np.fmin(steering_vsharpdegree, steering_verysharp)
activation_middle = np.fmin(steering_sharpdegree, steering_sharp)
activation_low = np.fmin(steering_milddegree, steering_medium)

plt.subplot(row, col, 5)
plt.title("Steering Activation: Mamdani Inference Method")
plt.plot(x_steering, activation_low, label="mild turn", marker=".")
plt.plot(x_steering, activation_middle, label="sharp turn", marker=".")
plt.plot(x_steering, activation_high, label="very sharp turn", marker=".")
plt.legend(loc="upper left")

# Apply the rules:
# * max for aggregation, like or the cases
aggregated = np.fmax(activation_low, np.fmax(activation_middle, activation_high))

# Defuzzification
steering_centroid = fuzz.defuzz(x_steering, aggregated, 'centroid')
steering_bisector = fuzz.defuzz(x_steering, aggregated, 'bisector')
steering_mom = fuzz.defuzz(x_steering, aggregated, "mom")
steering_som = fuzz.defuzz(x_steering, aggregated, "som")
steering_lom = fuzz.defuzz(x_steering, aggregated, "lom")

print(steering_centroid)
print(steering_bisector)
print(steering_mom)
print(steering_som)
print(steering_lom)

plt.subplot(row, col, 6)
plt.title("Aggregation and Defuzzification")
plt.plot(x_steering, aggregated, label="fuzzy result", marker=".")
plt.plot(steering_centroid, 0.0, label="centroid", marker="o")
plt.plot(steering_bisector, 0.0, label="bisector", marker="o")
plt.plot(steering_mom, 0.0, label="mom", marker="o")
plt.plot(steering_som, 0.0, label="som", marker="o")
plt.plot(steering_lom, 0.0, label="lom", marker="o")
plt.legend(loc="upper left")

# # ==================== FOR SPEED ===================================
# # Comment out the Steering section when graphing
# # for speed since the graph can hold a max of 6 plots.
# # use min since we have AND for the rule base
# # avoid obstacles when near
# speed_slowdegree = np.fmin(angle_small_degree, distance_near_degree)
# # avoid obstacles when medium
# speed_meddegree = np.fmin(angle_small_degree, distance_far_degree)
# # avoid obstacles when far
# speed_fastdegree = np.fmin(angle_medium_degree, distance_far_degree)
# # avoid obstacles when max away
# speed_maxdegree = np.fmin(angle_large_degree, distance_veryfar_degree)
#
# activation_slow = np.fmin(speed_slowdegree, speed_low)
# activation_medium = np.fmin(speed_meddegree, speed_medium)
# activation_fast = np.fmin(speed_fastdegree, speed_fast)
# activation_max = np.fmin(speed_maxdegree, speed_maximum)
#
# plt.subplot(row, col, 5)
# plt.title("Speed Activation: Mamdani Inference Method")
#
# plt.plot(x_speed, activation_slow, label="slow", marker=".")
# plt.plot(x_speed, activation_medium, label="medium", marker=".")
# plt.plot(x_speed, activation_fast, label="fast", marker=".")
# plt.plot(x_speed, activation_max, label="maximum", marker=".")
# plt.legend(loc="upper left")
#
# # Apply the rules:
# # * max for aggregation, like or the cases
# aggregated = np.fmax(activation_slow, np.fmax(activation_medium, activation_fast, activation_max))
#
# # Defuzzification
# speed_centroid = fuzz.defuzz(x_speed, aggregated, 'centroid')
# speed_bisector = fuzz.defuzz(x_speed, aggregated, 'bisector')
# speed_mom = fuzz.defuzz(x_speed, aggregated, "mom")
# speed_som = fuzz.defuzz(x_speed, aggregated, "som")
# speed_lom = fuzz.defuzz(x_speed, aggregated, "lom")
#
# print(speed_centroid)
# print(speed_bisector)
# print(speed_mom)
# print(speed_som)
# print(speed_lom)
#
# plt.subplot(row, col, 6)
# plt.title("Aggregation and Defuzzification")
# plt.plot(x_speed, aggregated, label="fuzzy result", marker=".")
# plt.plot(speed_centroid, 0.0, label="centroid", marker="o")
# plt.plot(speed_bisector, 0.0, label="bisector", marker="o")
# plt.plot(speed_mom, 0.0, label="mom", marker="o")
# plt.plot(speed_som, 0.0, label="som", marker="o")
# plt.plot(speed_lom, 0.0, label="lom", marker="o")
# plt.legend(loc="upper left")
#
