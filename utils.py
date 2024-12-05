"""
    @file        utils.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       Utilitaries functions for ML in Robot Kinematics
    @version     1.0
    @date        2024-12-04
    
"""

# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Robot parameters
l1 = 0.1  # First link
l2 = 0.1  # Second link
l3 = 0.1  # Third link



class Model:
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def get_name(self):
        return self.name


def displayLearningCurve(history, epochs: int):
    """
    Displays the model loss

    @param history: The model history
    @param epochs: The number of epochs
    """
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title("Model loss")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Test'], loc='upper right')

    plt.tight_layout()
    plt.show()


def FK(model: Model, theta: tuple):
    """
    Predicts the Forward Kinematics for a defined set of angles
    @param model: The machine learning model
    @param theta: The tuple of angle inputs
    """
    theta = tf.convert_to_tensor(theta, dtype=tf.float32) 
    n = theta.shape[0]
    if n == 2:    # 2R Robot
        features = tf.stack([tf.cos(theta[0]), tf.sin(theta[0]),
                             tf.cos(theta[1]), tf.sin(theta[1])])
    elif n == 3:  # 3R Robot
        features = tf.stack([tf.cos(theta[0]), tf.sin(theta[0]),
                             tf.cos(theta[1]), tf.sin(theta[1]),
                             tf.cos(theta[2]), tf.sin(theta[2])])
    elif n == 5:  # 5R Robot
        features = tf.stack([tf.cos(theta[0]), tf.sin(theta[0]),
                             tf.cos(theta[1]), tf.sin(theta[1]),
                             tf.cos(theta[2]), tf.sin(theta[2]),
                             tf.cos(theta[3]), tf.sin(theta[3]),
                             tf.cos(theta[4]), tf.sin(theta[4])])
    else:
        raise ValueError("Error! size of theta must be 2, 3, or 5!")
    
    # Reshape to batch size 1
    t = tf.reshape(features, shape=(1, features.shape[0]))
    out = model(t)
    
    # Reshape to the appropriate output vector
    output_shape = model.output_shape[1]
    out = tf.reshape(out, shape=(output_shape,))

    return out


def dispFK_2R(model: Model, theta: tuple, display_error: bool=False):
    """
    Plots the Forward Kinematics 2R comparison between Analytical method and ML model.

    model: The end-effector predicted position
    theta: The joint angles for the 2R robot
    display_error: Boolean to compute the relative error between analytical and ML
    """
    ee_x_pred, ee_y_pred = model 

    j0, j1 = theta
    
    # FK Analytical equations
    x1 = l1 * np.cos(j0)
    y1 = l1 * np.sin(j0)
    x2 = x1 + l2 * np.cos(j0 + j1)
    y2 = y1 + l2 * np.sin(j0 + j1)

    plt.figure(figsize=(5, 5))
    
    plt.plot([0, x1], [0, y1], 'c-', marker='o', label="Link 1 (Analytical)")
    plt.plot([x1, x2], [y1, y2], 'cyan', marker='o', label="Link 2 (Analytical)")

    # Plot ML-predicted end-effector
    plt.plot(ee_x_pred, ee_y_pred, 'mx', label="End-Effector (Model Prediction)")

    plt.xlim(-0.25, 0.25)
    plt.ylim(-0.25, 0.25)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2R Analytical and Model Prediciton Comparison")
    plt.grid()
    plt.legend()
    plt.show()

    if display_error:
        predicted_val = np.array([ee_x_pred, ee_y_pred])
        real_val = np.array([x2, y2])
        rel_error = np.abs(predicted_val-real_val)/np.abs(real_val)
        print(f"Relative error (%): {rel_error*100}")


def dispFK_3R(model: Model, theta: tuple, display_error: bool=False):
    """
    Plots the Forward Kinematics 3R comparison between Analytical method and ML model.

    model: The end-effector predicted position
    theta: The joint angles for the 3R robot
    display_error: Boolean to compute the relative error between analytical and ML
    """
    ee_x_pred, ee_y_pred = model 

    j0, j1, j2 = theta
    
    # FK Analytical equations
    x1 = l1 * np.cos(j0)
    y1 = l1 * np.sin(j0)
    x2 = x1 + l2 * np.cos(j0 + j1)
    y2 = y1 + l2 * np.sin(j0 + j1)
    x3 = l1*np.cos(j0) + l2 * np.cos(j0 + j1) + l3*np.cos(j0 + j1 + j2)
    y3 = l1*np.sin(j0) + l2 * np.sin(j0 + j1) + l3*np.sin(j0 + j1 + j2)

    plt.figure(figsize=(5, 5))
    
    plt.plot([0, x1], [0, y1], 'r-', marker='o', label="Link 1 (Analytical)")
    plt.plot([x1, x2], [y1, y2], 'orange', marker='o', label="Link 2 (Analytical)")
    plt.plot([x2, x3], [y2, y3], 'gold', marker='o', label="Link 3 (Analytical)")

    # Plot ML-predicted end-effector
    plt.plot(ee_x_pred, ee_y_pred, 'mx', label="End-Effector (Model Prediction)")

    plt.xlim(-0.35, 0.35)
    plt.ylim(-0.35, 0.35)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("3R Analytical and Model Prediciton Comparison")
    plt.grid()
    plt.legend()
    plt.show()

    if display_error:
        predicted_val = np.array([ee_x_pred, ee_y_pred])
        real_val = np.array([x3, y3])
        rel_error = np.abs(predicted_val-real_val)/np.abs(real_val)
        print(f"Relative error (%): {rel_error*100}")


@tf.function
def FK_Jacobian_pred(model: Model, x: tuple):
    """
    Computes the Forward Kinematics Jacobian matrix
    
    @param model: The machine learning model
    @param x: The tuple of angle inputs
    """
    theta = tf.convert_to_tensor(x, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(theta)
        y = FK(model, theta)
        n = theta.shape[0]
        output_shape = model.output_shape[0]

        # Removing the quaternion part for the Jacobian computation
        if n == 2 or n == 3:
            y = y if output_shape == 2 else y[:2]
        elif n == 5:
            y = y if output_shape == 3 else y[:3]
        else:
            raise ValueError("Error! size of theta must be 2, 3, or 5!")
        
    return tape.jacobian(y, theta)


def FK_Jacobian_analytic_2R(theta: tuple) -> np.ndarray:
    """
    Computes the Analytic Forward Kinematics Jacobian matrix
    @param theta: The tuple of angle inputs
    """
    j0, j1 = theta
    analytical_J = np.array([[-l1*np.sin(j0)-l2*np.sin(j0+j1), -l2*np.sin(j0+j1)],
                             [ l1*np.cos(j0)+l2*np.cos(j0+j1),  l2*np.cos(j0+j1)]])
    return analytical_J


def FK_Jacobian_analytic_3R(theta: float) -> np.ndarray:
    """
    Computes the Analytic Forward Kinematics Jacobian matrix for a 3R robot
    @param theta: The tuple of angle inputs
    """
    theta1, theta2, theta3 = theta

    analytical_J = np.array([
        [-l1*np.sin(theta1) - l2*np.sin(theta1 + theta2) - l3*np.sin(theta1 + theta2 + theta3),
         -l2*np.sin(theta1 + theta2) - l3*np.sin(theta1 + theta2 + theta3),
         -l3*np.sin(theta1 + theta2 + theta3)],
        
        [l1*np.cos(theta1) + l2*np.cos(theta1 + theta2) + l3*np.cos(theta1 + theta2 + theta3),
         l2*np.cos(theta1 + theta2) + l3*np.cos(theta1 + theta2 + theta3),
         l3*np.cos(theta1 + theta2 + theta3)]
    ])
    
    return analytical_J


def IK_Newton_Raphson(target_pos: tuple, initial_theta: tuple, model, max_iters=100, tolerance=1e-6) -> np.ndarray:
    '''
    Computes The Inverse Kinematics using the Newton-Raphson method for a 2R robot.

    @param target_pos: The target end-effector position 
    @param initial_theta: The initial value of joint angles
    @param model: The machine learning model
    @param max_iters: The maximum number of iterations
    @param tolerance: The tolerance for convergence
    '''
    theta = np.array(initial_theta, dtype=np.float32)
    for step in range(max_iters):
        current_pos = FK(model, theta)[:2]
        print(f"===== Step n°{step} =====")
        print(f"Current Position: {current_pos}")
        print(f"Intermediate joint angles: {theta}\n")

        error = target_pos - current_pos
        error = tf.reshape(error, (-1, 1))

        # Checking convergence
        if np.linalg.norm(error) < tolerance:
            break

        J = FK_Jacobian_pred(model, theta)

        dtheta = (tf.linalg.pinv(J) @ error)
        theta += dtheta.numpy().flatten()

    print(20*"=")
    print(f"Converegence reached in {step} steps")
    print(f"Computed joint angles for target position {target_pos}: (j0, j1) = {theta}")

    return theta


def dispIK_2R(model: Model, target_position: tuple, initial_guess: tuple, display_error: bool=False):
    """
    Plots the Inverse Kinematics 2R comparison between the initial and final positions

    model: The trained model for predicting end-effector position
    target_position: The desired end-effector position (x, y)
    initial_guess: Initial joint angles (j0, j1)
    display_error: Boolean to compute the relative error between analytical and ML
    """
    theta_solution = IK_Newton_Raphson(target_position, initial_guess, model)
    
    # Initial guess
    j0_initial, j1_initial = initial_guess
    x1_initial = l1 * np.cos(j0_initial)
    y1_initial = l1 * np.sin(j0_initial)
    x2_initial = x1_initial + l2 * np.cos(j0_initial + j1_initial)
    y2_initial = y1_initial + l2 * np.sin(j0_initial + j1_initial)

    # Final solution
    j0_final, j1_final = theta_solution
    x1_final = l1 * np.cos(j0_final)
    y1_final = l1 * np.sin(j0_final)
    x2_final = x1_final + l2 * np.cos(j0_final + j1_final)
    y2_final = y1_final + l2 * np.sin(j0_final + j1_final)

    # FK predicted end-effector position
    ee_pred = FK(model, theta_solution)
    ee_x_pred, ee_y_pred = ee_pred[:2]

    fig, axis = plt.subplots(1, 2, figsize=(12, 6))

    # Initial position
    axis[0].plot([0, x1_initial], [0, y1_initial], 'c-', marker='o', label="Link 1 (Initial)")
    axis[0].plot([x1_initial, x2_initial], [y1_initial, y2_initial], 'cyan', marker='o', label="Link 2 (Initial)")
    axis[0].plot(*target_position, 'rx', label="Target Position")
    axis[0].set_title("Initial Position")
    axis[0].set_xlabel('X')
    axis[0].set_ylabel('Y')
    axis[0].set_xlim(-0.25, 0.25)
    axis[0].set_ylim(-0.25, 0.25)
    axis[0].grid()
    axis[0].legend()

    # Final position
    axis[1].plot([0, x1_final], [0, y1_final], 'c-', marker='o', label="Link 1 (Final)")
    axis[1].plot([x1_final, x2_final], [y1_final, y2_final], 'cyan', marker='o', label="Link 2 (Final)")
    axis[1].plot(*target_position, 'rx', label="Target Position")
    axis[1].plot(ee_x_pred, ee_y_pred, 'mx', label="End-Effector (Model Prediction)")
    axis[1].set_title("Final Position")
    axis[1].set_xlabel('X')
    axis[1].set_ylabel('Y')
    axis[1].set_xlim(-0.25, 0.25)
    axis[1].set_ylim(-0.25, 0.25)
    axis[1].grid()
    axis[1].legend()

    plt.suptitle("Inverse Kinematics 2R: Initial vs Final Position")
    plt.show()

    if display_error:
        predicted_val = np.array([ee_x_pred, ee_y_pred])
        real_val = np.array([x2_final, y2_final])
        rel_error = np.abs(predicted_val-real_val)/np.abs(real_val)
        print(f"Relative error (%): {rel_error*100}")


def dispIK_3R(model: Model, target_position: tuple, initial_guess: tuple, display_error: bool=False):
    """
    Plots the Inverse Kinematics 3R comparison between the initial and final positions

    model: The trained model for predicting end-effector position
    target_position: The desired end-effector position (x, y)
    initial_guess: Initial joint angles (j0, j1, j2)
    display_error: Boolean to compute the relative error between analytical and ML
    """
    theta_solution = IK_Newton_Raphson(target_position, initial_guess, model)
    
    # Initial guess
    j0_initial, j1_initial, j2_initial = initial_guess
    x1_initial = l1 * np.cos(j0_initial)
    y1_initial = l1 * np.sin(j0_initial)
    x2_initial = x1_initial + l2 * np.cos(j0_initial + j1_initial)
    y2_initial = y1_initial + l2 * np.sin(j0_initial + j1_initial)
    x3_initial = x2_initial + l3 * np.cos(j0_initial + j1_initial + j2_initial)
    y3_initial = y2_initial + l3 * np.sin(j0_initial + j1_initial + j2_initial)

    # Final solution
    j0_final, j1_final, j2_final = theta_solution
    x1_final = l1 * np.cos(j0_final)
    y1_final = l1 * np.sin(j0_final)
    x2_final = x1_final + l2 * np.cos(j0_final + j1_final)
    y2_final = y1_final + l2 * np.sin(j0_final + j1_final)
    x3_final = x2_final + l3 * np.cos(j0_final + j1_final + j2_final)
    y3_final = y2_final + l3 * np.sin(j0_final + j1_final + j2_final)


    ee_pred = FK(model, theta_solution)
    ee_x_pred, ee_y_pred = ee_pred[:2]


    fig, axis = plt.subplots(1, 2, figsize=(14, 7))

    # Initial guess
    axis[0].plot([0, x1_initial], [0, y1_initial], 'r-', marker='o', label="Link 1 (Initial)")
    axis[0].plot([x1_initial, x2_initial], [y1_initial, y2_initial], 'orange', marker='o', label="Link 2 (Initial)")
    axis[0].plot([x2_initial, x3_initial], [y2_initial, y3_initial], 'gold', marker='o', label="Link 3 (Initial)")
    axis[0].plot(*target_position, 'cx', label="Target Position")
    axis[0].set_title("Initial Position")
    axis[0].set_xlabel('X')
    axis[0].set_ylabel('Y')
    axis[0].set_xlim(-0.35, 0.35)
    axis[0].set_ylim(-0.35, 0.35)
    axis[0].grid()
    axis[0].legend()

    # Final position
    axis[1].plot([0, x1_final], [0, y1_final], 'r-', marker='o', label="Link 1 (Final)")
    axis[1].plot([x1_final, x2_final], [y1_final, y2_final], 'orange', marker='o', label="Link 2 (Final)")
    axis[1].plot([x2_final, x3_final], [y2_final, y3_final], 'gold', marker='o', label="Link 3 (Final)")
    axis[1].plot(*target_position, 'cx', label="Target Position")
    axis[1].plot(ee_x_pred, ee_y_pred, 'mx', label="End-Effector (Model Prediction)")
    axis[1].set_title("Final Position")
    axis[1].set_xlabel('X')
    axis[1].set_ylabel('Y')
    axis[1].set_xlim(-0.35, 0.35)
    axis[1].set_ylim(-0.35, 0.35)
    axis[1].grid()
    axis[1].legend()

    plt.suptitle("Inverse Kinematics 3R: Initial vs Final Position")
    plt.show()

    if display_error:
        predicted_val = np.array([ee_x_pred, ee_y_pred])
        real_val = np.array([x3_final, y3_final])
        rel_error = np.abs(predicted_val-real_val)/np.abs(real_val)
        print(f"Relative error (%): {rel_error*100}")

