import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Coefficients from YAML file
positive_thrust_coefficients = [2.179239e+02, 2.997305e+00, -4.674627e-02, -3.968255e-01, 1.520276e-04]
negative_thrust_coefficients = [-2.946548e+02, -2.425470e+00, 3.983119e-02, 3.807154e-01, -1.103201e-04]

# Battery voltage
battery_voltage = 12  # Example battery voltage

# Generating PWM speeds for a range of thrusts
thrusts = np.linspace(-45, 60, 200)  # Example thrust range from -100 to 100
pwms = []

for thrust in thrusts:
    if thrust < 0:
        thrust_curve = Polynomial((
            negative_thrust_coefficients[0] + negative_thrust_coefficients[1] * battery_voltage +
            negative_thrust_coefficients[2] * battery_voltage ** 2 - thrust,
            negative_thrust_coefficients[3],
            negative_thrust_coefficients[4]
        ))
        roots = thrust_curve.roots()
        target_pwm_speed = roots.real[abs(roots.imag) < 1e-5][0]  # Taking the real root
    elif thrust > 0:
        thrust_curve = Polynomial((
            positive_thrust_coefficients[0] + positive_thrust_coefficients[1] * battery_voltage +
            positive_thrust_coefficients[2] * battery_voltage ** 2 - thrust,
            positive_thrust_coefficients[3],
            positive_thrust_coefficients[4]
        ))
        roots = thrust_curve.roots()
        target_pwm_speed = roots.real[abs(roots.imag) < 1e-5][1]  # Taking the real root
    else:
        target_pwm_speed = 1500  # Default PWM speed for zero thrust

    pwms.append(target_pwm_speed)

pwms = np.array(pwms)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(thrusts/6, pwms, label='PWM vs. Thrust')
plt.axhline(1500, color='red', linestyle='--', label='Default PWM Speed')
plt.xlabel('Thrust')
plt.ylabel('PWM Speed')
plt.title('PWM Speed vs. Thrust Curve')
plt.legend()
plt.grid(True)
plt.show()
