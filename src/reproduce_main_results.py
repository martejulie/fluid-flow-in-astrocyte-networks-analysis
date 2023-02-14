import dolfin as df
import os

# set path to solver
from ffian import flow_model, zero_flow_model
from plotter import Plotter


def run_model(model_v, j_in):
    """
    Arguments:
        model_v (str): model version
        j_in (float): constant input in input zone (mol/(m^2s))
    """

    # mesh
    N = 400                                  # mesh size
    L = 3.0e-4                               # m (300 um)
    mesh = df.IntervalMesh(N, 0, L)             # create mesh

    # time variables
    dt_value = 1e-3                          # time step (s)
    Tstop = 250                              # end time (s)

    # model setup
    t_PDE = df.Constant(0.0)                    # time constant

    if model_v == "M0":
        model = zero_flow_model.Model(mesh, L, t_PDE, j_in, stim_start=10, stim_end=210)
    else:
        model = flow_model.Model(model_v, mesh, L, t_PDE, j_in, stim_start=10, stim_end=210)

    # check that directory for results (data) exists, if not create
    path_data = '../results/data/' + model_v + '/'

    if not os.path.isdir(path_data):
        os.makedirs(path_data)

    # solve system
    if model_v == "M0":
        S = zero_flow_model.Solver(model, dt_value, Tstop)
    else:
        S = flow_model.Solver(model, dt_value, Tstop)

    S.solve_system(path_results=path_data)

    return model, path_data


if __name__ == '__main__':

    # run model setup M1
    model_v = 'M1'
    j_in = 8.0e-7
    model_M1, path_data_M1 = run_model(model_v, j_in)

    # run model setup M2
    model_v = 'M2'
    j_in = 9.15e-7
    model_M2, path_data_M2 = run_model(model_v, j_in)

    # run model setup M3
    model_v = 'M3'
    j_in = 9.05e-7
    model_M3, path_data_M3 = run_model(model_v, j_in)

    model_v = 'M0'
    j_in = 8.28e-7
    model_M4, path_data_M0 = run_model(model_v, j_in)

    # create plotter object for visualizing results
    P = Plotter(model_M1, path_data_M1, path_data_M2, path_data_M3, path_data_M0, verbose=True)

    # check that directory for figures exists, if not create
    path_figs = '../results/figures/'
    if not os.path.isdir(path_figs):
        os.makedirs(path_figs)

    # plot figures
    P.plot_model_dynamics(path_figs, 250)
    P.plot_flow_dynamics(path_figs, 200)
    P.plot_osmotic_pressures_and_water_potentials(path_figs, 200)
    P.plot_M2_and_M3_fluid_velocities(path_figs, 200)
    P.plot_ion_fluxes(path_figs, 200)
