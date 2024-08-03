---
layout: post
title:  "Particle Filter Implemented in Python"
date:   2024-08-03 00:00:00 +0000
tags: Particle Filter Implemented in Python Tutorial
color: rgb(200,200,200)
cover: '/assets/2024-08-02-ParticleFilter/particle.jpg'
# Particle Filter Implemented in Python 
---
 
# Particle  Filters implemented in Object tracking 
 
### What's Paricle filters?

Particle filter is a powerful technique used for state estimation, particularly effective in handling nonlinear systems, non-Gaussian distributions, and complex noise scenarios. It utilizes a large number of randomly sampled particles to represent the distribution of possible system states and dynamically updates their weights based on sensor measurements. Unlike traditional filters such as the Kalman filter, particle filter is not constrained by assumptions of linearity and Gaussianity, making it ideal for scenarios with multimodal distributions, unknown dynamic models, or nonlinear sensor measurements.

For example, in robot localization, particle filter excels in accurately tracking the robot's position and orientation amidst challenging environments and unpredictable motion patterns. This versatility makes particle filter a crucial tool in tasks like robot navigation, target tracking, and other dynamic system state estimations.

In this example, we will apply it into the tracking of dog's postion.
The original video:
<iframe type="text/html" width="100%" height="385" src="https://www.youtube.com/embed/1LYmxfMimBQ" frameborder="0"></iframe>

__The performance__:

The video features a running dog being tracked using a particle filter. The red box represents the position predicted by the particle with the highest weight, indicating the most likely position of the dog. The green box represents the weighted average position of all particles, providing an overall estimate by combining the predictions of all particles. This dual-box system helps in accurately tracking the dog's movement amidst challenging and dynamic environments.
<iframe type="text/html" width="100%" height="385" src="https://www.youtube.com/embed/bl6VRCHkETI" frameborder="0"></iframe>

__Steps of particle filter__

The core idea behind particle filtering is to use a large number of random samples (particles) to represent the distribution of possible states in a system, allowing for estimation and prediction of the system's state. To apply this algorithm, there are few steps to follow:

1. **Initialization**
  
    Randomly generate a set of particles and assign equal weights to them. These particles represent possible initial states of the system.
  
2. **Prediction**
  
    For each particle, use the system dynamics model and control inputs to predict the next state. This involves moving the particles according to the expected system behavior.

  
3. **Update (Weighting)**

    Update the weight of each particle based on the likelihood of the observed measurements given the particle's predicted state. Particles that better match the actual observations receive higher weights.

  
4. **Resample**

    Resample the particles based on their weights to focus on more probable states. Particles with higher weights are more likely to be duplicated, while particles with lower weights are likely to be discarded.

5. **Compute Estimation**
  
    Optionally, compute the state estimate from the weighted particles, typically using the weighted mean or mode of the particles.

6. **Loop**

   Repeat the prediction, update, resampling, and estimation steps for each new observation.

### Code Implementation ###


```python
import numpy as np
import cv2
```

### 1. State Prediction
The state of each particle is predicted based on a state transition model, which is often a linear model with added Gaussian noise.


```python
num_of_particles = 500
s_init = np.array([1600, 700, 0, 0])  # Initial state [x_center, y_center, x_velocity, y_velocity]

# For the first frame, all particles are initialized to the same state
s_new = np.tile(s_init, (num_of_particles, 1)).T
weights = np.ones(num_of_particles)
```

Where:

- ``s_init``represents the initial state of the target, including its center position and velocity.
- ``s_new`` initializes the particle states by replicating s_init.
- ``weights`` initializes all particle weights to 1.

### 2. Predict Particle States
Particles are predicted based on a state transition model. This model is linear and includes Gaussian noise to simulate process uncertainty.

$$
x_t^{(i)} = A \cdot x_{t-1}^{(i)} + v_t^{(i)}
$$

**Where:**
-  $x_t^{(i)}$  is the state of the \(i\)-th particle at time \(t\).
-  $A$  is the state transition matrix.
-  $v_t^{(i)}$ is the process noise for the \(i\)-th particle at time \(t\).



```python
def predict_particles(particles_states):
    dynamics = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    )
    predicted_states = dynamics @ particles_states + np.random.normal(loc=0, scale=1, size=particles_states.shape)
    return predicted_states
```

Where:
- ``dynamics`` is the state transition matrix A, which includes the transition model for position and velocity.
- ``np.random.normal`` adds Gaussian noise to the predicted states.


### 3. Compute Particle Weights
Weights are computed based on the likelihood of the observed data given the particle's state. This is done by comparing the histogram of the particle's state with the reference histogram.

Weight calculation formula:

$$
w_t^{(i)} \propto p(y_t | x_t^{(i)})
$$

**Where:**

 $$ p(y_t | x_t^{(i)}) $$ is the likelihood of the observation given the particle's state.


```python
def compute_weight(p, q):
    if not np.isclose(np.sum(p), 1):
        raise ValueError('p histogram is not normalized')
    if not np.isclose(np.sum(q), 1):
        raise ValueError('q histogram is not normalized')
    bc = np.sum(np.sqrt(p * q))  # Bhattacharyya coefficient
    return np.exp(20 * bc)
```

- Compute the histogram:


- Normalize the histogram:



### 4. Compute Normalized Histogram

Extract the region of interest (ROI) from the image based on the particle's state and compute its color histogram. Normalize the histogram for comparison.


```python
def compute_norm_hist(image, state):
    x_min = max(np.round(state[0] - half_width).astype(int), 0)
    x_max = min(np.round(state[0] + half_width).astype(int), image.shape[1] - 1)
    y_min = max(np.round(state[1] - half_height).astype(int), 0)
    y_max = min(np.round(state[1] + half_height).astype(int), image.shape[0] - 1)

    roi = image[y_min:y_max+1, x_min:x_max+1]
    roi_reduced = roi // 16
    roi_reduced = roi_reduced.astype(int)
    roi_indexing = (roi_reduced[..., 0] + roi_reduced[..., 1] * 16 + roi_reduced[..., 2] * 16 ** 2).flatten()
    hist, _ = np.histogram(roi_indexing, bins=4096, range=(0, 4096))
    norm_hist = hist / np.sum(hist)
    return norm_hist
```

Where:
- Extract the ROI based on the particle's state and the bounding box dimensions.
- Quantize color values to create a histogram hist.
- norm_hist is the normalized histogram.

### 5. Resample Particles
Resampling is done based on particle weights. This step ensures that particles with higher weights are more likely to be selected.
Resampling steps:
- Calculate the cumulative distribution function (CDF) of the weights.
- Use random numbers to sample indices according to the CDF.


```python
def sample_particle(particles_states, weights):
    normalized_weights = weights / np.sum(weights)
    sampling_weights = np.cumsum(normalized_weights)
    rand_numbers = np.random.random(sampling_weights.size)
    cross_diff = sampling_weights[None, :] - rand_numbers[:, None]
    cross_diff[cross_diff < 0] = np.inf
    sampling = np.argmin(cross_diff, axis=1)
    sampled_particles = particles_states[:, sampling]
    return sampled_particles
```

Where:
- sampling_weights is the cumulative sum of normalized weights.
- cross_diff is used to match random numbers with the cumulative weights to select particles.

### 6. Draw Bounding Box
Draw bounding boxes around the object based on the weighted average position and the position of the particle with the maximum weight.



```python
def draw_bounding_box(image, states, weights, with_max=True):
    mean_box = np.average(states, axis=1, weights=weights)
    x_c_mean, y_c_mean = np.round(mean_box[:2]).astype(int)
    image_with_boxes = image.copy()
    cv2.rectangle(image_with_boxes, (x_c_mean - half_width, y_c_mean - half_height),
                                     (x_c_mean + half_width, y_c_mean + half_height), (0, 255, 0), 1)
    if with_max:
        max_box = states[:, np.argmax(weights)]
        x_c_max, y_c_max = np.round(max_box[:2]).astype(int)
        cv2.rectangle(image_with_boxes, (x_c_max - half_width, y_c_max - half_height),
                                         (x_c_max + half_width, y_c_max + half_height), (0, 0, 255), 1)
    return image_with_boxes
```

Where:
- mean_box is the weighted average position of the particles.
- Draw bounding boxes for both the weighted average position and the particle with the maximum weight.

### 7.Draw Particles
Visualize particles on the image. The size of each particle is proportional to its weight.


```python
def draw_particles(image, states, weights):
    image_with_particles = image.copy()
    for s, w in zip(states.T, weights):
        x, y = np.round(s[:2]).astype(int)
        cv2.circle(image_with_particles, (x, y), int(round(30 * w)), (0, 0, 255), thickness=-1)
    return image_with_particles
```

Use cv2.circle to draw each particle, with the size proportional to the particle's weight.

### 8. Main Loop
Process each video frame iteratively, performing prediction, update, resampling, and visualization.

### Create Video Reader Object


```python
cap = cv2.VideoCapture('DSCF2822.MP4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit(1)
```

- Explanation: cv2.VideoCapture creates an object to read from the video file DSCF2822.MP4.
- Check: The if not cap.isOpened() block verifies that the video file was successfully opened. If not, it prints an error message and exits the program.

#### Read the First Frame


```python
ret, image = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit(1)
```

- Explanation: cap.read() reads the first frame from the video.
- Check: If reading the first frame fails (ret is False), it prints an error message and exits.

### Initialization of Number of Particles and Initial State


```python
num_of_particles = 500
s_init = np.array([1600, 700, 0, 0])  # initial state [x_center, y_center, x_velocity, y_velocity]
```

- set the number of particles used in the particle filter to 500. More particles generally provide better tracking accuracy but require more computation.
- s_init represents the initial state of the object being tracked. It is an array with four elements:x and y position and velocity.


Bounding Box Dimensions: 
These values define the dimensions of the bounding box around the object being tracked.


```python
half_height = 100
half_width = 80
```

Video Parameters


```python
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 20  # cap.get(5)  # frames per second
```

- frame_width and frame_height get the dimensions of the video frames from the cap object.
- fps sets the frames per second for the output video. Although commented out, it would typically be retrieved from the video file.

### Initialize the Video Writer


```python
size = (frame_width, frame_height)
result = cv2.VideoWriter('nini_tracked.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

if not result.isOpened():
    print("Error: Could not open video writer.")
    exit(1)
```

### Get the Normalized Histogram of the Initial State


```python
q = compute_norm_hist(image, s_init)
```

- Explanation:
 
``compute_norm_hist`` computes the histogram for the initial state s_init of the object in the first frame. 

This histogram serves as the reference for comparing with histograms of particles in subsequent frames.

### Initialize Particles and Weights


```python
s_new = np.tile(s_init, (num_of_particles, 1)).T
weights = np.ones(num_of_particles)
```

- s_new initializes the state of all particles to the initial state s_init. np.tile creates a matrix where each column is s_init, and .T transposes it to match the required shape.
- weights initializes the weight of each particle to 1, assuming equal probability for all particles initially.

## Main Loop
Processing frames:


```python
# go over the frames in the video
frame_count = 0
while True:
    # sample new particles according to the weights
    s_sampled = sample_particle(s_new, weights)
    # predict new particles (states) according to the previous states
    s_new = predict_particles(s_sampled)
    # go over the new predicted states, and compute the histogram for the state and
    # the weight with the original histogram
    for jj, s in enumerate(s_new.T):
        p = compute_norm_hist(image, s)
        weights[jj] = compute_weight(p, q)

    # draw bounding box over the tracked object
    image_with_boxes = draw_bounding_box(image, s_new, weights, with_max=True)
    result.write(image_with_boxes)

    # read next frame
    ret, image = cap.read()
    if not ret:  # if there is no next frame to read, stop
        print(f"End of video at frame {frame_count}.")
        break
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed frame {frame_count}.")

# release video objects
cap.release()
result.release()
print("Video processing completed and file saved.")

```
 
 