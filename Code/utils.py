from acq_functions import acq_functions
from scipy.optimize import minimize
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import functions


def optimise_acq_func(model, bounds, y_max, sample_count, acq_name="PI"):
    acq_object = acq_functions(acq_name, bounds, sample_count)
    x_max_val = acq_maximize(model, acq_object.acq_val, bounds, y_max)
    return x_max_val


# Optimizing the simple acq functions in acq_functions.py
def acq_maximize(gp, acq, bounds, y_max):
    x_tries = np.random.uniform(
        bounds[:, 0], bounds[:, 1], size=(1000, bounds.shape[0])
    )
    ys = acq(gp, x_tries, y_max)
    x_max = x_tries[np.random.choice(np.where(ys == ys.max())[0])]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(10, bounds.shape[0]))
    for x_try in x_seeds:
        res = minimize(
            lambda x: -acq(gp, x.reshape(1, -1), y_max),
            x_try.reshape(1, -1),
            bounds=bounds,
            method="L-BFGS-B",
        )
        if not res.success:
            continue
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking
    :param a: array to trim repeated rows from
    :return: mask of unique rows
    """
    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), "bool")
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


# Plot creation for 1-D functions
def plot_posterior_grad_1d(bounds, gp, X, Y, noise_val, count):
    if noise_val == True:
        noise = "Noisy "
    else:
        noise = "Noisless"
    t = np.linspace(-1, 15, 1000)
    objective = functions.cos()
    y = objective.func(t)
    mu, y_var = gp.predict(t[:, np.newaxis])
    std = np.sqrt(y_var)
    mu = np.squeeze(mu)
    std = np.squeeze(std)
    std = std.clip(min=0)
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(6, 3), dpi=100)
    plt.title(noise)
    plt.ylabel("Derivative Value")
    plt.xlabel("Input")
    ax1.plot(t, mu, label="Mean")
    ax1.fill_between(t, mu - 1.96 * std, mu + 1.96 * std, color="red", alpha=0.15)
    ax1.plot(X, Y, "ko", linewidth=2)
    ax1.plot(t, y, "b--", label="True value")
    ax1.legend(loc="lower right", frameon=False)
    filename = "Derivative_" + str(count) + "_" + noise + ".png"
    plt.savefig("Plots/" + filename)
    plt.show()


def plot_posterior_1d(bounds, gp, X, Y, noise_val, count):
    if noise_val == True:
        noise = "Noisy "
    else:
        noise = "Noisless"
    t = np.linspace(-1, 15, 1000)
    objective = functions.sin()
    y = objective.func(t)
    mu, y_var = gp.predict(t[:, np.newaxis])
    std = np.sqrt(y_var)
    mu = np.squeeze(mu)
    std = np.squeeze(std)
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(6, 3), dpi=100)
    plt.title(noise)
    plt.ylabel("Function Value")
    plt.xlabel("Input")
    ax1.plot(t, mu, label="Mean")
    ax1.fill_between(t, mu - 2 * std, mu + 2 * std, color="red", alpha=0.15)
    ax1.plot(X, Y, "ko", linewidth=2)
    ax1.plot(t, y, "b--", label="True value")
    ax1.legend(loc="lower right", frameon=False)
    filename = "Function_" + str(count) + "_" + noise + ".png"
    plt.savefig("Plots/" + filename)
    plt.show()


# Plot creation for 2-D functions (ploting the derivatives)
def plot_posterior_grad(bounds, gp_0, gp_1, X, Y, noise_val, count):
    if noise_val == True:
        noise = "Noisy "
    else:
        noise = "Noisless"
    # creating meshgrid to plot over entire range
    x1 = np.linspace(2, 10, 100)
    x2 = np.linspace(2, 10, 100)
    X1, X2 = np.meshgrid(x1, x2)
    t = np.vstack((X1.flatten(), X2.flatten())).T

    # mean and var for D=0 and D=1
    mu_0, y_var_0 = gp_0.predict(t)
    mu_1, y_var_1 = gp_1.predict(t)

    std_0 = np.sqrt(y_var_0)
    mu_0 = np.squeeze(mu_0)
    std_0 = np.squeeze(std_0)
    std_0 = std_0.clip(min=0)

    std_1 = np.sqrt(y_var_1)
    mu_1 = np.squeeze(mu_1)
    std_1 = np.squeeze(std_1)
    std_1 = std_1.clip(min=0)

    # The actual partial derivative of the function (used for comparision)
    #  out_0=np.cos(t[:,1])*np.cos(t[:,0])
    #  out_1=-np.sin(t[:,1])*np.sin(t[:,0])
    #  out_0=t[:,1]*np.cos(t[:,1])*(np.sin(t[:,0])+t[:,0]*np.cos(t[:,0]))
    #  out_1=t[:,0]*np.sin(t[:,0])*(np.cos(t[:,1])-t[:,1]*np.sin(t[:,1]))
    out_0 = (np.sin(t[:, 1]) * (2 * t[:, 0] * np.cos(t[:, 0]) - np.sin(t[:, 0]))) / (
        2 * t[:, 0] * np.sqrt(t[:, 0]) * np.sqrt(t[:, 1])
    )
    out_1 = (np.sin(t[:, 0]) * (2 * t[:, 1] * np.cos(t[:, 1]) - np.sin(t[:, 1]))) / (
        2 * t[:, 1] * np.sqrt(t[:, 0]) * np.sqrt(t[:, 1])
    )
    """ Define the Derivative function above yourself, looking into creating a function to calculate
    this"""
    # Ploting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2, ncols=2, figsize=(12, 6), dpi=100
    )
    im = ax2.pcolormesh(X1, X2, std_0.reshape(X1.shape), cmap="jet")
    fig.colorbar(im, ax=ax2)
    im1 = ax1.contour(X1, X2, out_0.reshape(X1.shape), cmap="PuBuGn")
    im2 = ax1.contour(X1, X2, mu_0.reshape(X1.shape), cmap="YlOrRd")
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax1)
    ax1.plot(X[:, 0], X[:, 1], "ok", markersize=5, alpha=0.8)
    ax1.title.set_text("Contour Plot of Mean D=0" + " " + noise)
    ax2.title.set_text("Standard deviation")

    im3 = ax4.pcolormesh(X1, X2, std_1.reshape(X1.shape), cmap="jet")
    fig.colorbar(im3, ax=ax4)
    im4 = ax3.contour(X1, X2, out_1.reshape(X1.shape), cmap="PuBuGn")
    im5 = ax3.contour(X1, X2, mu_1.reshape(X1.shape), cmap="YlOrRd")
    fig.colorbar(im4, ax=ax3)
    fig.colorbar(im5, ax=ax3)
    ax3.plot(X[:, 0], X[:, 1], "ok", markersize=5, alpha=0.8)
    ax3.title.set_text("Contour Plot of Mean D=1")
    ax4.title.set_text("Standard deviation")
    # Saving all the plots in 2D_Plots (need to creat a folder)
    filename = "SinxSiny_" + str(count) + "_" + noise + ".png"
    plt.savefig("2D_Plots/" + filename)
    plt.show()


def plot_posterior(bounds, gp, X, Y, count):
    noise = "Noisy"  # Noisy or Noiseless
    x1 = np.linspace(-4, 4, 100)
    x2 = np.linspace(-4, 4, 100)
    X1, X2 = np.meshgrid(x1, x2)
    t = np.vstack((X1.flatten(), X2.flatten())).T
    objective = functions.Kean()
    y = objective.func(t)
    mu, y_var = gp.predict(t)
    std = np.sqrt(y_var)
    mu = np.squeeze(mu)
    std = np.squeeze(std)
    fig = plt.figure(figsize=(13, 7))
    ax1 = fig.add_subplot(111, projection="3d")
    plt.title(noise)
    Ex, Ey = np.gradient(y.reshape(X1.shape))
    #  ax1.plot_surface(X1, X2, mu.reshape(X1.shape),cstride=1,
    #              cmap='viridis', edgecolor='none')
    #  ax1.scatter(X[:,0],X[:,1], Y, 'ko', linewidth=2,cmap='Reds')
    ax1.plot_wireframe(X1, X2, Ex, color="r")

    plt.show()


#  ax1.fill_between(t, mu-2*std, mu+2*std, color='red', alpha=0.15)
#  filename = 'Function_'+str(count)+'_'+noise+'.png'
#  plt.savefig('Plots/'+filename)
