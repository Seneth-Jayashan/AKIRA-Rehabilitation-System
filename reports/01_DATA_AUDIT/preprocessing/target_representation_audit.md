# Target Representation Audit

## TASK 1 — TARGET CONTINUITY AUDIT
- **Forearm Rotation:** The unnormalized target sequences demonstrate smooth temporal continuity, interrupted by exactly 13 catastrophic frame-to-frame jumps. 
  - *Example:* Window 2413 at frame 153 jumps from **-175.32° → 179.61°** (a difference of 354.93° in 8.3 milliseconds).
  - *Conclusion:* These are **mathematical angle wrapping** events caused by the optical solver crossing the ±180° Euler singularity, not real anatomical movements.
- **Elbow Angle:** The elbow angle exhibits no mathematical ±180° wrapping. We detected exactly 5 anomalous jumps (e.g., 144.31° → 6.05°), which correspond to optical marker occlusion/IK tracking loss, rather than coordinate wrapping.

## TASK 2 — ANGLE WRAP ANALYSIS
Applying mathematical unwrapping (`numpy.unwrap`) completely eliminates the 350°+ jumps, restoring a continuous physical trajectory. 
- **Wrapped:** Maximum step change = ~359.62°.
- **Unwrapped:** Maximum step change = < 5.0°, accurately reflecting physiological angular velocity limits.

## TASK 3 — DETERMINE APPROPRIATE TARGET REPRESENTATION
### Option A: Direct angle regression in degrees
- **Advantages:** Native interpretability.
- **Disadvantages:** MSE computes the loss between `179°` and `-179°` as `358²`, punishing the model catastrophically for being physically correct but mathematically wrapped. 

### Option B: Unwrapped angle regression
- **Advantages:** Eliminates the wrapping penalty; allows standard MSE.
- **Disadvantages:** If a sensor drifts globally over multiple rotations, the absolute reference is lost. 

### Option C: Sine/Cosine representation (`sin(θ), cos(θ)`)
- **Advantages:** Strictly respects the $S^1$ circular topology of rotational angles. The distance between `179°` and `-179°` is infinitesimally small in sin/cos space. Loss perfectly scales with physical orientation error.
- **Disadvantages:** Doubles the output dimension for that joint; requires `atan2` post-processing to calculate ROM and angular velocity during deployment.

**Decision:** The most scientifically defensible representation for a rotating frame subject to Euler wrapping is **Option C (Sine/Cosine)**.

## TASK 4 — ELBOW ANGLE AUDIT
Unlike the forearm, the elbow acts as a 1-DoF hinge physically constrained between ~0° (extension) and ~150° (flexion). It never physically traverses 360°, nor does its reference frame force it across a ±180° singularity in this dataset.
**Decision:** Direct regression (Option A) perfectly represents the elbow's limited linear topology.

## TASK 5 — CLASSIFICATION TARGET AUDIT
- **Number of Classes:** 20 distinct task/exercise classes.
- **Subject Coverage:** 19 out of 20 classes are present in exactly 15 subjects. The remaining class (`WristDevCali`) is present in 14 subjects.
- **Conclusion:** Every class has exceptional subject coverage, fully supporting robust Subject-Independent evaluation.

## TASK 6 — MULTI-TASK LEARNING FEASIBILITY
The dataset robustly supports a shared representation for all three tasks:
1. Every window possesses rich 12-channel IMU inputs.
2. Every window is continuously labeled with optical kinematic ground truth.
3. Every window belongs to a fully represented classification hierarchy.
There is zero label sparsity preventing simultaneous multi-task backpropagation.

---

## TASK 7 — RECOMMEND THE FINAL MODEL OUTPUT

### 1. Forearm representation recommendation
**Sine/Cosine Vector.** The network should predict `[sin(forearm_θ), cos(forearm_θ)]` to correctly model the wrapped circular domain without artificial penalty gradients.

### 2. Elbow representation recommendation
**Direct Angle Regression.** The network should predict `[elbow_θ]` directly (optionally scaled to `[0, 1]` or standard normal space during intermediate layers, but returning a linear continuous scalar).

### 3. Classification recommendation
**Softmax Logits.** A 20-dimensional classification head for the exercise labels.

### 4. Final TCN Output Structure
**Input:** `(Batch, 240, 12)`
**Outputs:**
- `y_elbow`: `(Batch, 240, 1)`  *(Continuous sequence)*
- `y_forearm`: `(Batch, 240, 2)` *(Continuous sequence of sin/cos)*
- `y_exercise`: `(Batch, 20)` *(Window-level classification)*

### 5. Recommended Loss Functions
- **Elbow:** Mean Squared Error (MSE) or Huber Loss (to resist the 5 tracking anomalies).
- **Forearm:** Mean Squared Error (MSE) applied jointly to the `[sin, cos]` vector, or explicit Cosine Distance loss.
- **Exercise:** Categorical Cross-Entropy (CCE).

### 6. Remaining Risks
- The 5 identified anomalous jumps in the elbow target (due to tracking loss) will introduce massive, incorrect gradients if standard MSE is used. **Huber Loss** is strongly recommended for the elbow regression head to bound the gradient explosion during these corrupted frames.
