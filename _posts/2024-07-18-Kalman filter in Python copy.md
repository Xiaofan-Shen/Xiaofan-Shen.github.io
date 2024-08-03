---
layout: post
title:  "Kalman Filter Implemented in Python"
date:   2024-07-18 00:00:00 +0000
tags: KalmanFilter StateEstimation Tutorial
color: rgb(200,200,200)
cover: '/assets/2024-07-18-KalmanFilter/Kalman_filter_cover.png'
# Kalman Filter Implemented in Python 
---

# Kalman Filter

In many real-world applications, such as tracking the position of a moving vehicle, we rely on models to predict the state of a system. However, these predictions are never perfect due to uncertainties and noise in the system. To improve the accuracy of our state estimates, we use a method called the Kalman filtering.

The Kalman filter is an efficient recursive algorithm that estimates the state of a dynamic system from a series of incomplete and noisy measurements. 


**Modeling a Dynamic System**

The dynamics of the system can be represented by the following equations:
- The model of the system without noise is:

$$
\dot{\mathbf x} = \mathbf{Ax}
$$

- The model of the system with noise is:

$$ \dot{\mathbf x} = \mathbf{Ax} + \mathbf w$$

Consider any inputs into the system. We assume an input $\mathbf u$, and a linear model matrix $\mathbf B$ is set to convert $\mathbf u$ into the effect on the system.  

- Including Inputs:

$$ \dot{\mathbf x} = \mathbf{Ax} + \mathbf{Bu} + \mathbf{w}$$


Where： 

- $\dot{\mathbf x}$ is time derivative of the state $\mathbf x$
, representing the rate of change of the system's state.

- $\mathbf{A}$ is the system matrix, which describes the relationship between the state variables.

- $\mathbf{B}$  is the input matrix, which describes how the input vector $\mathbf{u}$  affects the state variables.

- $\mathbf{u}$  is the input vector, containing the control inputs to the system.

- $\mathbf{w}$  is the disturbance vector, representing external disturbances or noise affecting the system.

These equations form the foundation for modeling dynamic systems, allowing engineers to predict and analyze the system's behavior under varying conditions and inputs.

**State Transition Equation** 

While continuous-time dynamic models provide theoretical insights into how system states change over time, practical applications demand more specific and actionable models to predict and manage system behaviors at specific moments. This is where the introduction of the state transition equation becomes crucial.

The state transition equation describes how the state of a system evolves over discrete time steps:

$$\mathbf x_k = \mathbf{Fx}_{k-1} + \mathbf B_k\mathbf u_k$$



Where:

- ${\mathbf F}$ is the state transition matrix, which has the ability to transition the state's value between discrete time steps.

- $\mathbf{B_k}$ is the control input matrix, which specifies how the control input influences the state transition at time k. It allows flexibility in influence of external inputs in the system.

- $\mathbf{u_k}$ is the control input vector at time k. It represents external inputs or commands applied to the system at time, which affect the evolution of the state from k-1 to k.


This equation provides a framework for modeling the temporal evolution of a dynamic system under the influence of external inputs or controls. By adjusting 
F and B, engineers can simulate and analyze how different inputs affect the system's behavior over time.

## Python Implementation of Kalman Filter

In this case, we use an simple 1D position and velocity example without external control inputs, which system's state is determined solely by its own dynamics and process noise:

$$\mathbf x_k = \mathbf{Fx}_{k-1}$$


#### Step 1 Given offsers of each variables in the state vector

These defines the order of the state variables in our vector $\mathbf x$


```python
import numpy as np
# offsets of each variable in the state vector
iX = 0
iV = 1
NUMVARS = iV + 1  
```

- 'iX' and 'iV' are the indices for position and velocity in the state vector, respectively. 
    - in this case, we have 
    $$\mathbf x = \begin{bmatrix}x&v\end{bmatrix}^T$$, where $x$ is position, and $v$ is the velocity
- 'NUMVARS' represents the number of variables in the state vector, which is 2 in this case (position and velocity)
    - this will be used to initialise the size of our arrays.

#### Step 2 Initialize the class object instance:


```python
class KF:
    def __init__(self, initial_x: float, initial_v: float, accel_variance: float) -> None:
        # mean of state GRV
        self._x = np.zeros(NUMVARS)
        self._x[iX] = initial_x 
        self._x[iV] = initial_v  
        self._accel_variance = accel_variance
        # covariance of state GRV
        self._P = np.eye(NUMVARS)

```

init initializes the Kalman filter instance variables:
- self._x is the state vector $\mathbf x$, including position and velocity, initialized to a zero vector and then set to the initial position and velocity.
- self._accel_variance is the variance of the acceleration, describing the process noise in the system.
- self._P is the state covariance matrix, initialized to an identity matrix, representing the initial covariance between position and velocity.

#### Step 3 Prediction: state prediction and covariance prediction


```python
def predict(self, dt: float) -> None:
    F = np.eye(NUMVARS)
    F[iX, iV] = dt 
    new_x = F.dot(self._x) 
    
    G = np.zeros((2, 1))
    G[iX] = 0.5 * dt**2
    G[iV] = dt 
    new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance

    self._P = new_P
    self._x = new_x
```

**1. State prediction**

F is the state transition matrix, we have position and velocity to track, and to describe how the state evolves over a time step dt, $F$ is: 

$$
\begin{aligned}
\mathbf F &= \begin{bmatrix}1&0\\0&1\end{bmatrix} + \begin{bmatrix}0&1\\0&0\end{bmatrix}\Delta t\\
&= \begin{bmatrix}1&\Delta t\\0&1\end{bmatrix}
\end{aligned}$$

Apply this into $\mathbf x_k= \mathbf{Fx}_{k-1}$ to get the predict new_x, which is:

$$
\begin{aligned}
\mathbf x_k &=\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix} \mathbf x_{k-1}
\end{aligned}$$



**2. Covariance prediction:**
In addition to predicting the state, it's essential to predict the covariance because the covariance matrix describes the uncertainty of the state estimation. We denote the predicted covariance matrix as $P_{k+1}$. There are two components in this part:

1. **Impact of State Prediction on Covariance:**

    During the state prediction (predict) phase, the current state covariance matrix is updated to the predicted state covariance matrix using the state transition matrix F and the process noise covariance matrix 𝑄. This step accounts for the uncertainty in state changes that the system model cannot fully capture.

2. **Contribution of Process Noise:**

    The process noise covariance matrix Q plays a crucial role in the state prediction by describing the unmodeled dynamic changes in the system and the influence of control inputs. It directly affects the size and structure of the predicted state covariance matrix.
    
Based on these two factors, we predict the new state covariance matrix \( P \):

$$
\mathbf{P}_{k+1} = F \mathbf{P}_k F^T + Q 
$$

Recall that Q describes the unmodeled dynamic changes in the system and the influence of control inputs, so it involves G and $\sigma_a^2$:

$$
\mathbf{Q} = G G^T \sigma^2_a 
$$

To sum up:

$$
\mathbf{P}_{k+1} = F \mathbf{P}_k F^T + G G^T \sigma^2_a 
$$

where  $\sigma^2_a $ is the variance of the acceleration noise.


**G**
This prediction involves two main components:
For an object moving with constant acceleration a, the position change over a time interval dt can be expressed as:
$$ x = x_0 + v_0 \cdot dt + \frac{1}{2} a \cdot dt^2 $$

So that, `G[iX] = 0.5 * dt^2` and `G[iV] = dt` indicate the influence of acceleration noise on position and velocity over the time step $t$ and $G$ matrix is:
The process noise influence matrix \( $G$ \) is:

$$

   G = \begin{pmatrix} 
0.5 \cdot dt^2 \\ 
dt 
\end{pmatrix} 
$$

- Process Noise Influence Matrix $G$:
- $G$ is the matrix describing the influence of process noise on the system state. Its size matches the state vector.
- The $G$ matrix considers the impact of acceleration noise on both position and velocity.
- `G[iX] = 0.5 * dt^2` and `G[iV] = dt` indicate the influence of acceleration noise on position and velocity over the time step t.


Using the G matrix, we predict the new state covariance matrix \( P \):

$$
\mathbf{P}_{k+1} = F \mathbf{P}_k F^T + G G^T \sigma^2_a 
$$

where  $\sigma^2_a $ is the variance of the acceleration noise.

#### Step 4 Update

The update step is crucial because it allows the Kalman filter to correct its predictions based on new measurements. Imagine you have a system (like a moving car) and you're trying to estimate its position and speed. You use a model to predict where the car will be, but your prediction won't be perfect due to uncertainties (like changes in speed or direction).

Every time you get a new measurement (like a GPS reading), you can use this new information to correct your estimate. This makes your estimation more accurate than just relying on predictions alone.

The update step of the Kalman filter utilizes linear algebra and probability theory to dynamically adjust the state estimate based on real-time measurement data. These mathematical formulas ensure that the filter optimally balances between the system dynamics and measurement uncertainties, providing accurate and reliable state estimation.

Thus, the update step involves correcting the predicted state estimate based on the new measurement. Here's a detailed explanation of the update process:

1. **Calculate Measurement Residual:**

   Calculate the measurement residual $y$ , which is the difference between the actual measurement $z$  and the predicted measurement based on the current state estimate 
   
   $$ y = z - H \mathbf{x} $$

   where:
   -  $\mathbf{z}$ is the measured measurement.
   -  $\mathbf{H}$ is the observation matrix that maps the state vector into the measurement space.
   -  $\mathbf{x}$  is the predicted state vector.


2. **Calculate Residual Covariance:**

   Compute the residual covariance $S$ , which combines the prediction uncertainty and measurement noise:

  $$ S = H \mathbf{P} H^T + R $$

   where:
   -  $\mathbf{P}$   is the predicted state covariance matrix.
   -  $\mathbf{R}$   is the covariance matrix of the measurement noise.


3. **Compute Kalman Gain:**

    The Kalman Gain is a factor that determines how much you should adjust your prediction based on the new measurement. It balances the uncertainty in your prediction against the uncertainty in the measurement. If your prediction is very uncertain but your measurement is very accurate, the Kalman Gain will be higher, meaning you trust the measurement more.
   Calculate the Kalman gain $ K $, which determines how much the predicted state estimate is adjusted based on the measurement residual:
   
   $$ K = \mathbf{P} H^T (H \mathbf{P} H^T + R)^{-1} $$


4. **Update State Estimate:**

   Update the state estimate  $\mathbf{x}$  using the Kalman gain and the measurement residual:
   
    $$\mathbf{x}^+ = \mathbf{x} + K y $$
    
   Adjust your previous state estimate by adding the product of the Kalman Gain and the residual. This step corrects the prediction by bringing it closer to the actual measurement.


5. **Update Covariance Matrix:**

   Update the covariance matrix $ \mathbf{P} $ to reflect the incorporation of new measurement information:
   
   $$ \mathbf{P}^+ = (I - K H) \mathbf{P} $$

### Significance in Applications

The update step enables the Kalman filter to adaptively adjust the state estimate according to real-world measurement data, facilitating precise tracking and prediction of system states. This capability makes the Kalman filter invaluable in real-time applications such as navigation systems, robotics, and sensor data processing.


**Implementation of Update in Python:**


```python
def update(self, meas_value: float, meas_variance: float):
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P

        H = np.array([1, 0]).reshape((1, 2))

        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))
        
        new_X = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._P = new_P
        self._x = new_X
```

## Using the KF 
Now let's test the kalman filter implementation

We now create the `main.py` script to demonstrates how to use the Kalman Filter to track the position and velocity of an object over time.To better observe the performance of the Kalman Filter, we need to set up the plotting environment and initialize various parameters:

**1. Import Libraries and Modules and Set Plotting Environment:**

Import the necessary libraries and modules. numpy is used for numerical computations, matplotlib for plotting, and KF is the Kalman Filter implementation.


```python
import numpy as np
import matplotlib.pyplot as plt
from kf import KF 
```

**2. Initialization and Parameter Setup**

Initializing the real position and velocity of the object is necessary to simulate the actual state of the system we want to track. This provides a reference to compare against the filter's estimates.


```python
real_x = 0.0
meas_variance = 0.1 ** 2
real_v = 0.5

kf = KF(initial_x = 0.0, initial_v = 1.0, accel_variance = 0.1)
```

**3. Simulation Parameters**

Defining the noise in the measurements simulates real-world sensor inaccuracies. This helps in testing the filter's ability to handle and correct noisy data.


```python
DT = 0.1
NUM_STEPS = 1000
MEAS_EVERY_STEPS = 20

mus = []
covs = []
real_xs = []
real_vs = []
```

- `DT` is the time step, controlling the interval at which the simulation updates.
- `NUM_STEPS` is the total number of steps to simulate, determining the duration of the simulation.
- `MEAS_EVERY_STEPS` defines how often we take measurements, simulating periodic sensor readings.
- `mus`, `covs`, `real_xs`, and `real_vs` are lists to store the filter's estimates, covariances, and the true positions and velocities for later analysis and plotting.

**4. Simulation Loop**

To emulate the dynamic behavior of the system over time and to observe how the Kalman Filter performs in real-time tracking of the object's state, we use simulation loop.  Here’s a detailed explanation of why we need the simulation loop:
- **Dynamic Behavior Simulation:** 
    In real-world applications, the state of the system (such as the position and velocity of an object) changes continuously over time. The simulation loop allows us to replicate this dynamic behavior in a controlled environment, updating the state of the system at each time step.
- **State Prediction and Update:**
    The Kalman Filter operates in two main steps: prediction and update. The simulation loop enables these steps to be executed repeatedly.
- **Performance Evaluation:**
    By running the simulation loop over multiple iterations, we can observe how well the Kalman Filter tracks the system's state over time. This helps in evaluating the filter’s accuracy and responsiveness to changes in the system.


```python
for step in range(NUM_STEPS):
    if step > 500:
        real_v *= 0.9

    covs.append(kf.cov)
    mus.append(kf.mean)
    
    real_x = real_x + DT * real_v

    kf.predict(dt = DT)
    if step != 0 and step % MEAS_EVERY_STEPS == 0:
        kf.update(meas_value = real_x + np.random.randn() * np.sqrt(meas_variance), 
                  meas_variance = meas_variance)

    real_xs.append(real_x)
    real_vs.append(real_v)
```

**5. Plotting**

To visually assess the Kalman Filter's performance in estimating dynamic system states from noisy measurements, plotting is indispensable. It provides a clear comparison between estimated and actual states, visualizes uncertainties, aids in performance evaluation, and facilitates debugging and optimization of the filter implementation. 

Apart from the estimated and actual states, we also plot the Confidence Interval of the estimated position/velocity (mean ± 2 standard deviations), which helps us understand the range within which the true velocity is likely to lie with a high degree of confidence. It reflects the uncertainty in our velocity estimation due to measurement variations and model assumptions. Plotting this interval alongside the estimated velocity (mu[1]) allows us to visualize the accuracy and reliability of our velocity predictions throughout the simulation.


```python
plt.subplot(2, 1, 1)
plt.title('Position')
plt.plot([mu[0] for mu in mus], 'g')
plt.plot(real_xs, 'b')
plt.plot([mu[0] - 2*np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')
plt.plot([mu[0] + 2*np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')

plt.subplot(2, 1, 2)
plt.title('Velocity')
plt.plot([mu[1] for mu in mus], 'g')
plt.plot(real_vs, 'b')
plt.plot([mu[1] - 2*np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--')
plt.plot([mu[1] + 2*np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--')
plt.show()
```


    
<img src="\assets\2024-07-18-KalmanFilter\kalman_filter.png" alt="Description of the image" width="500" height="320">
    

- **The first subplot (top) titled 'Position' displays：**

    1. Green solid line: Estimated position `mu[0]` from the Kalman filter.
    2. Blue solid line: Real position `real_xs`.
    3. Red dashed lines: Confidence interval of the estimated position `mean ± 2 standard deviations`.

- **The second subplot (bottom) titled 'Velocity' displays:**
    1. Green solid line: Estimated position `mu[0]` from the Kalman filter.
    2. Blue solid line: Real velocity `real_vs`.
    3. Red dashed lines: Confidence interval of the estimated velocity `mean ± 2 standard deviations`.

**Performance Evaluation:**

- **Position:**

    The predicted positions closely match the true values, indicating excellent performance of the Kalman filter in tracking the system's position. Additionally, the confidence interval for the estimated velocity is narrow, suggesting high confidence in the velocity estimates and minimal deviation from the actual velocity values. However, there may be noticeable fluctuations near zero, possibly due to inaccuracies in the system model when the position is near stationary or at rest.
    
- **Velocity:**

    Regarding velocity, the predicted values closely approximate the true values overall. However, during changes in velocity, there are noticeable discrepancies. The confidence interval is broader compared to position, indicating lower confidence in velocity estimates and larger potential deviations from actual values. Similar to position, there are significant fluctuations around zero, which are more pronounced for velocity estimates than for position estimates.

**Conclusion**


The Kalman filter demonstrates strong performance in tracking the position of the object, with predictions closely matching the true values. The confidence interval for position estimation remains narrow, indicating high confidence in these predictions, except for fluctuations near zero where variability increases.    For velocity, while the Kalman filter accurately tracks changes, there are some deviations observed during rapid changes. The confidence interval for velocity estimates is high, suggesting lower uncertainty in velocity predictions. Same to the position prediction, around zero the fluctuations are more pronounced.  Overall, the Kalman filter effectively reduces noise and provides reliable state estimation for dynamic systems.








