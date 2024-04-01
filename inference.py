from timeit import default_timer as timer


def generate_sample(dae, diffusion, cvae, num_samples, enable_autocast, ode_eps, ode_solver_tol,
                    temp=1.0, noise=None, c=None):
    start = timer()
    z, nfe, time_ode_solve = diffusion.sample_model_ode(dae, num_samples, ode_eps, ode_solver_tol,
                                                        enable_autocast, temp, noise)
    x = cvae.decode(z, c)
    y = cvae.predict(z, c)
    end = timer()
    sampling_time = end - start
    return x, y, nfe, sampling_time
