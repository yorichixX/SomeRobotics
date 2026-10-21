# Mathematical Derivation: Epipolar Geometry & Disparity-to-Depth Relation

## 1. Overview and Stereo Geometry Setup

Consider a calibrated two-camera stereo vision system in a canonical rectified configuration:
- Both cameras have identical focal length $f$ and principal points $(c_x, c_y)$.
- The optical axes of both cameras are strictly parallel and aligned along the $Z$-axis.
- The baseline separation between the two camera optical centers $O_L$ (left camera) and $O_R$ (right camera) is $B$ along the horizontal $X$-axis.

```
       Left Camera                          Right Camera
          O_L                                   O_R
           | <--------------- Baseline B --------------> |
           |                                             |
           |                                             |
       f   |                                         f   |
           v                                             v
     [Image Plane L]                               [Image Plane R]
           |                                             |
           x_L                                           x_R
            \                                           /
             \                                         /
              \                                       /
               \                                     /
                \                                   /
                 \                                 /
                  \                               /
                   \                             /
                    \                           /
                     ------> 3D Point P <-------
                             (X, Y, Z)
```

---

## 2. Geometric Coordinate Projections

Let a 3D real-world point $P$ have coordinates in the left camera coordinate frame:

$$
P_L = \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = [X, Y, Z]^T
$$

Because the right camera is displaced purely horizontally along the positive $X$-axis by baseline distance $B$:

$$
P_R = \begin{bmatrix} X - B \\ Y \\ Z \end{bmatrix} = [X - B, Y, Z]^T
$$

Using the ideal pinhole camera projection equations:

### Left Camera Image Projection:

$$
x_L = f \cdot \frac{X}{Z} + c_x
$$

$$
y_L = f \cdot \frac{Y}{Z} + c_y
$$

### Right Camera Image Projection:

$$
x_R = f \cdot \frac{X - B}{Z} + c_x
$$

$$
y_R = f \cdot \frac{Y}{Z} + c_y
$$

Notice that for rectified cameras, the vertical coordinates are identical ($y_L = y_R$). Therefore, conjugate search is strictly 1D along the horizontal scanline.

---

## 3. Derivation of the Disparity Equation

Disparity $d$ is defined as the horizontal pixel difference between corresponding image points:

$$
d \triangleq x_L - x_R
$$

Substitute the projection equations:

$$
d = \left( f \cdot \frac{X}{Z} + c_x \right) - \left( f \cdot \frac{X - B}{Z} + c_x \right)
$$

Cancel the principal point offsets $c_x$:

$$
d = f \cdot \frac{X}{Z} - f \cdot \frac{X - B}{Z}
$$

Factor out $\frac{f}{Z}$:

$$
d = \frac{f}{Z} \left( X - (X - B) \right)
$$

$$
d = \frac{f \cdot B}{Z}
$$

---

## 4. Derivation of Depth from Disparity

Rearranging the disparity equation yields the fundamental **Disparity-to-Depth Relation**:

$$
\boxed{Z = \frac{f \cdot B}{d}}
$$

Where:
- $Z$ = Orthogonal depth / distance from camera baseline to point $P$ (meters).
- $f$ = Camera focal length (pixels).
- $B$ = Physical baseline distance between left and right camera centers (meters).
- $d$ = Measured horizontal disparity between corresponding image features (pixels).

---

## 5. 3D World Coordinate Reconstruction

Once depth $Z$ is calculated, the full 3D coordinates $(X, Y, Z)$ of point $P$ in the left camera frame can be resolved:

$$
X = \frac{(x_L - c_x) \cdot Z}{f} = \frac{(x_L - c_x) \cdot B}{d}
$$

$$
Y = \frac{(y_L - c_y) \cdot Z}{f} = \frac{(y_L - c_y) \cdot B}{d}
$$

$$
Z = \frac{f \cdot B}{d}
$$

---

## 6. Error Propagation and Sensitivity Analysis

To understand measurement uncertainty on edge robotics systems, take the first derivative of depth $Z$ with respect to disparity $d$:

$$
\frac{\partial Z}{\partial d} = \frac{\partial}{\partial d} \left( \frac{f \cdot B}{d} \right) = -\frac{f \cdot B}{d^2}
$$

Substitute $d = \frac{f \cdot B}{Z}$ into the denominator:

$$
\frac{\partial Z}{\partial d} = -\frac{f \cdot B}{\left(\frac{f \cdot B}{Z}\right)^2} = -\frac{Z^2}{f \cdot B}
$$

Expressing finite measurement error:

$$
\boxed{|\Delta Z| \approx \frac{Z^2}{f \cdot B} \cdot |\Delta d|}
$$

### Critical Engineering Insights for Robotics:
1. **Quadratic Error Growth:** Depth estimation error $|\Delta Z|$ increases with the **square of the distance** ($Z^2$).
2. **Sub-pixel Disparity Requirement:** For objects further than $20\text{ m}$, even a $0.5\text{ px}$ disparity matching error causes significant depth uncertainty.
3. **Baseline Trade-off:** Increasing baseline $B$ reduces depth error proportionally, but increases the minimum blind-spot distance in front of the robot bumper.

---

## 7. The Epipolar Constraint: Essential and Fundamental Matrices

In general unrectified stereo vision, the relationship between two views is governed by Epipolar Geometry:

Let $x_L$ and $x_R$ be homogeneous 2D coordinates in left and right image frames.
Let $R$ and $t$ be the relative rotation matrix and translation vector from Left to Right camera.

### The Essential Matrix $E$:

$$
E = [t]_\times R
$$

where $[t]_\times$ is the skew-symmetric cross-product matrix:

$$
[t]_\times = \begin{bmatrix} 0 & -t_z & t_y \\ t_z & 0 & -t_x \\ -t_y & t_x & 0 \end{bmatrix}
$$

For normalized coordinates $\hat{x}_L = K_L^{-1} x_L$ and $\hat{x}_R = K_R^{-1} x_R$, the coplanarity condition of vectors $O_L P$, $O_R P$, and $O_L O_R$ enforces:

$$
\hat{x}_R^T E \hat{x}_L = 0
$$

### The Fundamental Matrix $F$:

Transforming back to unnormalized pixel coordinates:

$$
x_R^T K_R^{-T} E K_L^{-1} x_L = 0
$$

$$
\boxed{x_R^T F x_L = 0}
$$

where:

$$
F = K_R^{-T} E K_L^{-1}
$$

### Epipolar Line Interpretation:

For any point $x_L$ in the left camera, its corresponding point in the right camera must lie along the 1D epipolar line:

$$
l_R = F x_L
$$

Rectification mathematically warps both image planes so that all epipolar lines become strictly horizontal and parallel to image rows ($l_R = [0, 1, -y_L]^T$), which reduces 2D stereo correspondence matching to an ultra-fast 1D horizontal search line along identical scanlines ($y_L = y_R$).
