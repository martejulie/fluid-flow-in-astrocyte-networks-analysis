import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import dolfin as df

# set font & text parameters
font = {'family': 'serif',
        'weight': 'bold',
        'size': 16}
plt.rc('font', **font)
plt.rc('text', usetex=True)
mpl.rcParams['image.cmap'] = 'jet'
plt.rc('legend')
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')

# set colors
colormap = cm.viridis
mus = [1, 2, 3, 4, 5, 6]
colorparams = mus
colormap = cm.viridis
normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))

b0 = '#faee00ff'
b1 = '#7171ffff'
b2 = '#feb608ff'

c2 = colormap(normalize(mus[0]))
c1 = colormap(normalize(mus[1]))
c0 = colormap(normalize(mus[2]))
c3 = colormap(normalize(mus[3]))
c4 = colormap(normalize(mus[4]))
c5 = colormap(normalize(mus[5]))
colors = [c1, c2, c3, c4, c5]

# plotting parameters
xlim = [0, 3e-4]  # range of x values (m)
xticks = [0e-3, 0.05e-3, 0.1e-3, 0.15e-3, 0.2e-3, 0.25e-3, 0.3e-3]
xticklabels = ['0', '50', '100', '150', '200', '250', '300']
xlabel_x = '$x$ (um)'
point_time = 1.5e-4

lw = 4.5     # line width
fosi = 18.7  # ylabel font size
fs = 0.9


class Plotter():

    def __init__(self, model, path_data_I=None, path_data_II=None, path_data_III=None, path_data_0=None, verbose=False):

        self.model = model
        self.verbose = verbose

        N_ions = self.model.N_ions
        N_comparts = self.model.N_comparts
        self.N_unknowns = N_comparts*(1 + N_ions) + 3

        # initialize mesh
        self.mesh_PDE = df.Mesh()

        # initialize data files
        if path_data_I is not None:
            self.set_datafile_I(path_data_I)
        if path_data_II is not None:
            self.set_datafile_II(path_data_II)
        if path_data_III is not None:
            self.set_datafile_III(path_data_III)
        if path_data_0 is not None:
            self.set_datafile_0(path_data_0)

        return

    def set_datafile_I(self, path_data):
        # file containing data
        self.h5_fname_PDE_I = path_data + 'PDE/results.h5'

        # create mesh and read data file
        self.hdf5_PDE_I = df.HDF5File(df.MPI.comm_world, self.h5_fname_PDE_I, 'r')
        self.hdf5_PDE_I.read(self.mesh_PDE, '/mesh', False)

        return

    def set_datafile_II(self, path_data):
        # file containing data
        self.h5_fname_PDE_II = path_data + 'PDE/results.h5'

        # read data file
        self.hdf5_PDE_II = df.HDF5File(df.MPI.comm_world, self.h5_fname_PDE_II, 'r')
        self.hdf5_PDE_II.read(self.mesh_PDE, '/mesh', False)

        return

    def set_datafile_III(self, path_data):
        # file containing data
        self.h5_fname_PDE_III = path_data + 'PDE/results.h5'

        # read data file
        self.hdf5_PDE_III = df.HDF5File(df.MPI.comm_world, self.h5_fname_PDE_III, 'r')
        self.hdf5_PDE_III.read(self.mesh_PDE, '/mesh', False)

        return

    def set_datafile_0(self, path_data):
        # file containing data
        self.h5_fname_PDE_0 = path_data + 'PDE/results.h5'

        # read data file
        self.hdf5_PDE_0 = df.HDF5File(df.MPI.comm_world, self.h5_fname_PDE_0, 'r')
        self.hdf5_PDE_0.read(self.mesh_PDE, '/mesh', False)

        return

    def read_from_file_I(self, n, i):
        """ get snapshot of solution w[i] at time = n seconds """

        N_comparts = self.model.N_comparts
        N_unknowns = self.N_unknowns

        CG1 = df.FiniteElement('CG', self.mesh_PDE.ufl_cell(), 1)
        R = df.FiniteElement("R", self.mesh_PDE.ufl_cell(), 0)  # element for Lagrange multiplier
        e = [CG1]*(N_comparts - 1) + [CG1]*(N_unknowns - (N_comparts - 1))
        W = df.FunctionSpace(self.mesh_PDE, df.MixedElement(e + [R]))
        u = df.Function(W)

        V_CG1 = df.FunctionSpace(self.mesh_PDE, CG1)
        f = df.Function(V_CG1)

        self.hdf5_PDE_I.read(u, "/solution/vector_" + str(n))
        df.assign(f, u.split()[i])

        return f

    def read_from_file_II(self, n, i):
        """ get snapshot of solution w[i] at time = n seconds """

        N_comparts = self.model.N_comparts
        N_unknowns = self.N_unknowns

        CG1 = df.FiniteElement('CG', self.mesh_PDE.ufl_cell(), 1)
        R = df.FiniteElement("R", self.mesh_PDE.ufl_cell(), 0)  # element for Lagrange multiplier
        e = [CG1]*(N_comparts - 1) + [CG1]*(N_unknowns - (N_comparts - 1))
        W = df.FunctionSpace(self.mesh_PDE, df.MixedElement(e + [R]))
        u = df.Function(W)

        V_CG1 = df.FunctionSpace(self.mesh_PDE, CG1)
        f = df.Function(V_CG1)

        self.hdf5_PDE_II.read(u, "/solution/vector_" + str(n))
        df.assign(f, u.split()[i])

        return f

    def read_from_file_III(self, n, i):
        """ get snapshot of solution w[i] at time = n seconds """

        N_comparts = self.model.N_comparts
        N_unknowns = self.N_unknowns

        CG1 = df.FiniteElement('CG', self.mesh_PDE.ufl_cell(), 1)
        R = df.FiniteElement("R", self.mesh_PDE.ufl_cell(), 0)  # element for Lagrange multiplier
        e = [CG1]*(N_comparts - 1) + [CG1]*(N_unknowns - (N_comparts - 1))
        W = df.FunctionSpace(self.mesh_PDE, df.MixedElement(e + [R]))
        u = df.Function(W)

        V_CG1 = df.FunctionSpace(self.mesh_PDE, CG1)
        f = df.Function(V_CG1)

        self.hdf5_PDE_III.read(u, "/solution/vector_" + str(n))
        df.assign(f, u.split()[i])

        return f

    def read_from_file_0(self, n, i):
        """ get snapshot of solution w[i] at time = n seconds """

        N_comparts = self.model.N_comparts
        N_unknowns = self.N_unknowns

        CG1 = df.FiniteElement('CG', self.mesh_PDE.ufl_cell(), 1)
        R = df.FiniteElement("R", self.mesh_PDE.ufl_cell(), 0)  # element for Lagrange multiplier
        e = [CG1]*(N_comparts - 1) + [CG1]*(N_unknowns - (N_comparts - 1))
        W = df.FunctionSpace(self.mesh_PDE, df.MixedElement(e + [R]))
        u = df.Function(W)

        V_CG1 = df.FunctionSpace(self.mesh_PDE, CG1)
        f = df.Function(V_CG1)

        self.hdf5_PDE_0.read(u, "/solution/vector_" + str(n))
        df.assign(f, u.split()[i])

        return f

    def project_to_function_space(self, u):
        """ project u onto function space """

        CG1 = df.FiniteElement('CG', self.mesh_PDE.ufl_cell(), 1)
        V = df.FunctionSpace(self.mesh_PDE, CG1)
        f = df.project(u, V)

        return f

    def plot_model_dynamics(self, path_figs, Tstop):
        """ Plot input/decay-currents, changes in ECS and ICS ion concentrations,
        changes in ECS and ICS volume fractions,
        changes in transmembrane hydrostatic pressure,
        and membrane potential over time, measured at x = point_time. """

        # get parameters
        K_m = self.model.params['K_m']
        p_m_init = self.model.params['p_m_init']
        alpha_i_init = float(self.model.alpha_i_init)
        alpha_e_init = float(self.model.alpha_e_init)

        # point in space at which to use in timeplots
        point = point_time

        # range of t values
        xlim_T = [0.0, Tstop]

        # list of function values at point
        j_ins = []
        j_decs = []
        Na_is = []
        K_is = []
        Cl_is = []
        Na_es = []
        K_es = []
        Cl_es = []
        dalpha_is = []
        dalpha_es = []
        p_ms = []
        phi_ms = []

        for n in range(Tstop+1):

            # get data
            alpha_i = self.read_from_file_I(n, 0)
            Na_i = self.read_from_file_I(n, 1)
            Na_e = self.read_from_file_I(n, 2)
            K_i = self.read_from_file_I(n, 3)
            K_e = self.read_from_file_I(n, 4)
            Cl_i = self.read_from_file_I(n, 5)
            Cl_e = self.read_from_file_I(n, 6)
            phi_i = self.read_from_file_I(n, 7)
            phi_e = self.read_from_file_I(n, 8)

            # calculate extracellular volume fraction
            alpha_e = 0.6 - alpha_i(point)

            # get input/decay fluxes
            j_in_ = self.model.j_in(n)
            j_dec_ = self.model.j_dec(K_e)
            j_in = self.project_to_function_space(j_in_*1e6)    # convert to umol/(m^2s)
            j_dec = self.project_to_function_space(j_dec_*1e6)  # convert to umol/(m^2s)

            # calculate change in volume fractions
            alpha_i_diff = (alpha_i(point) - alpha_i_init)/alpha_i_init*100
            alpha_e_diff = (alpha_e - alpha_e_init)/alpha_e_init*100

            # calculate transmembrane hydrostatic pressure
            tau = K_m*(alpha_i(point) - alpha_i_init)
            p_m = tau + p_m_init

            # calculate membrane potential
            phi_m = (phi_i(point) - phi_e(point))*1000  # convert to mV

            # append data to lists
            j_ins.append(j_in(point))
            j_decs.append(j_dec(point))
            Na_is.append(Na_i(point))
            K_is.append(K_i(point))
            Cl_is.append(Cl_i(point))
            Na_es.append(Na_e(point))
            K_es.append(K_e(point))
            Cl_es.append(Cl_e(point))
            dalpha_is.append(alpha_i_diff)
            dalpha_es.append(alpha_e_diff)
            p_ms.append(float(p_m))
            phi_ms.append(phi_m)

        # create plot
        fig = plt.figure(figsize=(11*fs, 15*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(4, 2, 1, xlim=xlim_T, ylim=[-0.25, 1.0])
        plt.ylabel(r'$j\mathrm{^K_{input}}$($\mu$mol/(m$^2$s))', fontsize=fosi)
        plt.plot(j_ins, color='k', linestyle='dotted', linewidth=lw)

        ax2 = fig.add_subplot(4, 2, 2, xlim=xlim_T, ylim=[-0.25, 1.0])
        plt.ylabel(r'$j\mathrm{^K_{decay}}$($\mu$mol/(m$^2$s))', fontsize=fosi)
        plt.plot(j_decs, color='k', linewidth=lw)

        ax3 = fig.add_subplot(4, 2, 3, xlim=xlim_T, ylim=[-20, 10])
        plt.ylabel(r'$\Delta [k]_\mathrm{e}$ (mM)', fontsize=fosi)
        plt.plot(np.array(Na_es)-Na_es[0], color=b0, label=r'Na$^+$', linewidth=lw)
        plt.plot(np.array(K_es)-K_es[0], color=b1, label=r'K$^+$', linestyle='dotted', linewidth=lw)
        plt.plot(np.array(Cl_es)-Cl_es[0], color=b2, label=r'Cl$^-$', linestyle='dashed', linewidth=lw)

        ax4 = fig.add_subplot(4, 2, 4, xlim=xlim_T, ylim=[-20, 10])
        plt.ylabel(r'$\Delta [k]_\mathrm{i}$ (mM)', fontsize=fosi)
        plt.plot(np.array(Na_is)-Na_is[0], color=b0, linewidth=lw)
        plt.plot(np.array(K_is)-K_is[0], color=b1, linestyle='dotted', linewidth=lw)
        plt.plot(np.array(Cl_is)-Cl_is[0], color=b2, linestyle='dashed', linewidth=lw)

        ax5 = fig.add_subplot(4, 2, 5, xlim=xlim_T, ylim=[-30, 1])
        plt.ylabel(r'$\Delta \alpha_\mathrm{e}$ (\%) ', fontsize=fosi)
        plt.plot(dalpha_es, color=c0, linewidth=lw)

        ax6 = fig.add_subplot(4, 2, 6, xlim=xlim_T, ylim=[-1, 30])
        plt.ylabel(r'$\Delta \alpha_\mathrm{i}$ (\%) ', fontsize=fosi)
        plt.plot(dalpha_is, color=c0, linewidth=lw)

        ax7 = fig.add_subplot(4, 2, 7, xlim=xlim_T)
        plt.ylabel(r'$\Delta(p_\mathrm{i}- p_\mathrm{e})$ (Pa)', fontsize=fosi)
        plt.plot(np.array(p_ms)-float(p_m_init), color=c2, linewidth=lw)
        plt.xlabel(r'time (s)', fontsize=fosi)

        ax8 = fig.add_subplot(4, 2, 8, xlim=xlim_T, ylim=[-90, -60])
        plt.ylabel(r'$\phi_\mathrm{m}$ (mV)', fontsize=fosi)
        plt.plot(phi_ms, color=c1, linewidth=lw)
        plt.xlabel(r'time (s)', fontsize=fosi)

        plt.figlegend(bbox_to_anchor=(0.35, 0.68), frameon=True)

        # make pretty
        ax.axis('off')

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}',
                   r'\textbf{C}', r'\textbf{D}',
                   r'\textbf{E}', r'\textbf{F}',
                   r'\textbf{G}', r'\textbf{H}']
        for num, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]):
            ax.text(-0.2, 1.0, letters[num], transform=ax.transAxes, size=22, weight='bold')
            # make pretty
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        plt.tight_layout()

        # save figure to file
        fname_res = path_figs + 'Figure2'
        plt.savefig(fname_res + '.pdf', format='pdf')
        plt.close()

        if self.verbose:

            print('Max K_e increase:', max(np.array(K_es)-K_es[0]))
            print('Max Na_e decrease:', min(np.array(Na_es)-Na_es[0]))
            print('Max Cl_e decrease:', min(np.array(Cl_es)-Cl_es[0]))

            print('Max K_i increase:', max(np.array(K_is)-K_is[0]))
            print('Max Na_i decrease:', min(np.array(Na_is)-Na_is[0]))
            print('Max Cl_i increase:', max(np.array(Cl_is)-Cl_is[0]))
            print('dK_i in steady-state:', np.array(K_is)[round(len(K_is)/2)]-K_is[0])

            print('max alpha_i increase:', max(np.array(dalpha_is)))
            print('max alpha_e decrease:', min(np.array(dalpha_es)))
            print('max dp incrase:', max(np.array(p_ms)-float(p_m_init)))
            print('initial phi_m:', phi_ms[0])
            print('max phi_m:', max(phi_ms))

        return

    def plot_flow_dynamics(self, path_figs, n):
        """ Plot transmembrane- and compartmental pressures
        (osmotic and hydrostatic), and transmembrane- and
        compartmental fluid velocities at t = n sec. """

        # get parameters
        temperature = self.model.params['temperature']
        R = self.model.params['R']
        K_m = self.model.params['K_m']
        eta_m = self.model.params['eta_m']
        kappa = self.model.params['kappa']
        p_m_init = self.model.params['p_m_init']
        a_i = self.model.params['a'][0]
        a_e = self.model.params['a'][1]
        alpha_i_init = float(self.model.alpha_i_init)

        # get data at t = 0
        alpha_i_0 = self.read_from_file_I(0, 0)
        Na_i_0 = self.read_from_file_I(0, 1)
        Na_e_0 = self.read_from_file_I(0, 2)
        K_i_0 = self.read_from_file_I(0, 3)
        K_e_0 = self.read_from_file_I(0, 4)
        Cl_i_0 = self.read_from_file_I(0, 5)
        Cl_e_0 = self.read_from_file_I(0, 6)
        p_e_0 = self.read_from_file_I(0, 9)

        # extracellular volume fraction at t = 0
        alpha_e_0 = 0.6 - alpha_i_0

        # intracellular hydrostatic pressure at t = 0 (Pa)
        tau_0 = K_m*(alpha_i_0 - alpha_i_init)
        p_i_0 = p_e_0 + tau_0 + p_m_init

        # osmotic concentrations at t = 0 (mM)
        Osm_i_0 = Na_i_0 + K_i_0 + Cl_i_0 + a_i/alpha_i_0
        Osm_e_0 = Na_e_0 + K_e_0 + Cl_e_0 + a_e/alpha_e_0

        # solute potentials at t = 0 (Pa)
        Pi_i_0 = -R*temperature*(Osm_i_0)
        Pi_e_0 = -R*temperature*(Osm_e_0)

        # get data at t = n
        alpha_i = self.read_from_file_I(n, 0)
        Na_i = self.read_from_file_I(n, 1)
        Na_e = self.read_from_file_I(n, 2)
        K_i = self.read_from_file_I(n, 3)
        K_e = self.read_from_file_I(n, 4)
        Cl_i = self.read_from_file_I(n, 5)
        Cl_e = self.read_from_file_I(n, 6)
        p_e = self.read_from_file_I(n, 9)

        # extracellular volume fraction
        alpha_e = 0.6 - alpha_i

        # intracellular hydrostatic pressure
        tau = K_m*(alpha_i - alpha_i_init)
        p_i = p_e + tau + p_m_init

        # osmotic concentrations at t = n (mM)
        Osm_i = Na_i + K_i + Cl_i + a_i/alpha_i
        Osm_e = Na_e + K_e + Cl_e + a_e/alpha_e

        # solute potentials at t = n (Pa)
        Pi_i = -R*temperature*(Osm_i)
        Pi_e = -R*temperature*(Osm_e)

        # changes from baseline
        dOsm_i = Osm_i - Osm_i_0
        dOsm_e = Osm_e - Osm_e_0
        dPi_m = (Pi_i - Pi_e) - (Pi_i_0 - Pi_e_0)
        dp_m = p_i - p_e - (p_i_0 - p_e_0)
        dp_i = p_i - p_i_0
        dp_e = p_e - p_e_0

        # transmembrane water flux
        w_m_ = p_i - p_e
        w_m_ += R*temperature*(a_e/alpha_e - a_i/alpha_i + Na_e - Na_i + K_e - K_i + Cl_e - Cl_i)
        w_m = eta_m*w_m_*1.0e6*60  # convert to um/min

        # ICS fluid velocities
        u_i = - alpha_i*kappa[0]*df.grad(p_i)*1.0e6*60  # convert to um/min

        # ECS fluid velocities
        u_e = - alpha_e*kappa[1]*df.grad(p_e)*1.0e6*60  # convert to um/min

        # project to function space
        dOsm_i = self.project_to_function_space(dOsm_i)
        dOsm_e = self.project_to_function_space(dOsm_e)
        dPi_m = self.project_to_function_space(dPi_m)
        w_m = self.project_to_function_space(w_m)
        dp_i = self.project_to_function_space(dp_i)
        dp_e = self.project_to_function_space(dp_e)
        dp_m = self.project_to_function_space(dp_m)
        u_i = self.project_to_function_space(u_i[0])
        u_e = self.project_to_function_space(u_e[0])

        # create plot
        fig = plt.figure(figsize=(15*fs, 10*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(2, 3, 1, xlim=xlim)
        plt.xticks(xticks, xticklabels)
        plt.ylabel(r'$\Delta$O (mM)')
        df.plot(dOsm_e, label=r'ECS', color=colors[2], linewidth=lw)
        df.plot(dOsm_i, label=r'ICS', color='k', linewidth=lw, linestyle='dotted')

        ax2 = fig.add_subplot(2, 3, 2, xlim=xlim, ylim=[-800, 200])
        plt.xticks(xticks, xticklabels)
        plt.ylabel(r'$\Delta(\Pi_\mathrm{i}-\Pi_\mathrm{e})$ and $\Delta(p_\mathrm{i}-p_\mathrm{e})$ (Pa)')
        df.plot(dPi_m, label=r'$\Delta(\Pi_\mathrm{i}-\Pi_\mathrm{e})$', color=c1, linewidth=lw)
        df.plot(dp_m, label=r'$\Delta(p_\mathrm{i}-p_\mathrm{e})$', color=c2,
                linewidth=lw, linestyle=(0, (3, 1, 1, 1, 1, 1)))

        ax3 = fig.add_subplot(2, 3, 3, xlim=xlim, ylim=[-0.003, 0.001])
        plt.xticks(xticks, xticklabels)
        plt.ylabel(r'$w_\mathrm{m}$ ($\mu$m/min)')
        df.plot(w_m, color=c0, linewidth=lw, linestyle=(0, (5, 1)))

        ax4 = fig.add_subplot(2, 3, 4, xlim=xlim, ylim=[-100, 30])
        plt.xticks(xticks, xticklabels)
        plt.xlabel(xlabel_x, fontsize=fosi)
        plt.ylabel(r'$\Delta p$ (Pa)')
        df.plot(dp_e, color=colors[2], linewidth=lw)
        df.plot(dp_i, color='k', linewidth=lw, linestyle='dotted')

        ax5 = fig.add_subplot(2, 3, 5, xlim=xlim, ylim=[-0.35, 0.35])
        plt.xticks(xticks, xticklabels)
        plt.xlabel(xlabel_x, fontsize=fosi)
        plt.ylabel(r'$\alpha u$ ($\mu$m/min)')
        df.plot(u_e, color=colors[2], linewidth=lw)
        df.plot(u_i, color='k', linewidth=lw, linestyle='dotted')

        ax6 = fig.add_subplot(2, 3, 6)

        plt.figlegend(bbox_to_anchor=(0.9, 0.4), frameon=True)

        # make pretty
        ax.axis('off')

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}', r'\textbf{E}', r'\textbf{F}']
        for num, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
            ax.text(-0.25, 1.0, letters[num], transform=ax.transAxes, size=22, weight='bold')
            # make pretty
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        ax6.set_axis_off()
        plt.tight_layout()

        # save figure to file
        fname_res = path_figs + 'Figure3'
        plt.savefig(fname_res + '.pdf', format='pdf')
        plt.close()

        if self.verbose:
            print('Max O_i decrease:', min(dOsm_i.vector()[:]))
            print('Mmax O_e increase:', min(dOsm_e.vector()[:]))
            print('Max P_i-P_e decrease:', min(dPi_m.vector()[:]))
            print('Max w_m:', min(w_m.vector()[:]))
            print('Max p_i increase:', max(dp_i.vector()[:]))
            print('Max p_e decrease:', min(dp_e.vector()[:]))
            print('Min u_i:', min(u_i.vector()[:]))
            print('Min u_e:', min(u_e.vector()[:]))
            print('Max u_i:', max(u_i.vector()[:]))
            print('Max u_e:', max(u_e.vector()[:]))

        return

    def plot_osmotic_pressures_and_water_potentials(self, path_figs, n):
        """ Plot changes in ECS and ICS osmotic concentrations,
        changes in osmotic pressures, changes in ECS solute potentials,
        changes in ECS hydrostatic pressures and changes in ECS
        water potentials at t = n. """

        # get parameters
        temperature = self.model.params['temperature']
        R = self.model.params['R']
        a_i = self.model.params['a'][0]
        a_e = self.model.params['a'][1]

        # get data at t = 0
        alpha_i_0 = self.read_from_file_I(0, 0)
        Na_i_0 = self.read_from_file_I(0, 1)
        Na_e_0 = self.read_from_file_I(0, 2)
        K_i_0 = self.read_from_file_I(0, 3)
        K_e_0 = self.read_from_file_I(0, 4)
        Cl_i_0 = self.read_from_file_I(0, 5)
        Cl_e_0 = self.read_from_file_I(0, 6)
        p_e_0 = self.read_from_file_I(0, 9)
        alpha_e_0 = 0.6 - alpha_i_0

        # osmotic concentrations at t = 0 (mM)
        Osm_i_0 = Na_i_0 + K_i_0 + Cl_i_0 + a_i/alpha_i_0
        Osm_e_0 = Na_e_0 + K_e_0 + Cl_e_0 + a_e/alpha_e_0

        # solute potentials at t = 0 (Pa)
        Pi_i_0 = -R*temperature*Osm_i_0
        Pi_e_0 = -R*temperature*Osm_e_0

        # get data M1
        alpha_i_M1 = self.read_from_file_I(n, 0)
        Na_i_M1 = self.read_from_file_I(n, 1)
        Na_e_M1 = self.read_from_file_I(n, 2)
        K_i_M1 = self.read_from_file_I(n, 3)
        K_e_M1 = self.read_from_file_I(n, 4)
        Cl_i_M1 = self.read_from_file_I(n, 5)
        Cl_e_M1 = self.read_from_file_I(n, 6)
        p_e_M1 = self.read_from_file_I(n, 9)
        alpha_e_M1 = 0.6 - alpha_i_M1

        # change in p_e from baseline
        dp_e_M1 = p_e_M1 - p_e_0

        # M1 osmotic concentrations (mM)
        Osm_i_M1 = Na_i_M1 + K_i_M1 + Cl_i_M1 + a_i/alpha_i_M1
        Osm_e_M1 = Na_e_M1 + K_e_M1 + Cl_e_M1 + a_e/alpha_e_M1
        # change from baseline
        dOsm_i_M1 = Osm_i_M1 - Osm_i_0
        dOsm_e_M1 = Osm_e_M1 - Osm_e_0

        # M1 solute potentials (Pa)
        Pi_i_M1 = -R*temperature*(Osm_i_M1)
        Pi_e_M1 = -R*temperature*(Osm_e_M1)
        # change from baseline
        dPi_i_M1 = Pi_i_M1 - Pi_i_0
        dPi_e_M1 = Pi_e_M1 - Pi_e_0

        # get data M2
        alpha_i_M2 = self.read_from_file_II(n, 0)
        Na_i_M2 = self.read_from_file_II(n, 1)
        Na_e_M2 = self.read_from_file_II(n, 2)
        K_i_M2 = self.read_from_file_II(n, 3)
        K_e_M2 = self.read_from_file_II(n, 4)
        Cl_i_M2 = self.read_from_file_II(n, 5)
        Cl_e_M2 = self.read_from_file_II(n, 6)
        p_e_M2 = self.read_from_file_II(n, 9)
        alpha_e_M2 = 0.6 - alpha_i_M2

        # change in p_e from baseline
        dp_e_M2 = p_e_M2 - p_e_0

        # M2 osmotic concentrations (mM)
        Osm_i_M2 = Na_i_M2 + K_i_M2 + Cl_i_M2 + a_i/alpha_i_M2
        Osm_e_M2 = Na_e_M2 + K_e_M2 + Cl_e_M2 + a_e/alpha_e_M2
        # change from baseline
        dOsm_i_M2 = Osm_i_M2 - Osm_i_0
        dOsm_e_M2 = Osm_e_M2 - Osm_e_0

        # M2 solute potentials (Pa)
        Pi_i_M2 = -R*temperature*(Osm_i_M2)
        Pi_e_M2 = -R*temperature*(Osm_e_M2)
        # change from baseline
        dPi_i_M2 = Pi_i_M2 - Pi_i_0
        dPi_e_M2 = Pi_e_M2 - Pi_e_0

        # get data M0
        Na_i_M0 = self.read_from_file_0(n, 0)
        Na_e_M0 = self.read_from_file_0(n, 1)
        K_i_M0 = self.read_from_file_0(n, 2)
        K_e_M0 = self.read_from_file_0(n, 3)
        Cl_i_M0 = self.read_from_file_0(n, 4)
        Cl_e_M0 = self.read_from_file_0(n, 5)
        alpha_i_M0 = 0.4
        alpha_e_M0 = 0.2

        # M0 osmotic concentrations (mM)
        Osm_i_M0 = Na_i_M0 + K_i_M0 + Cl_i_M0 + a_i/alpha_i_M0
        Osm_e_M0 = Na_e_M0 + K_e_M0 + Cl_e_M0 + a_e/alpha_e_M0
        # change from baseline
        dOsm_i_M0 = Osm_i_M0 - Osm_i_0
        dOsm_e_M0 = Osm_e_M0 - Osm_e_0

        # M0 solute potentials (Pa)
        Pi_i_M0 = -R*temperature*(Osm_i_M0)
        Pi_e_M0 = -R*temperature*(Osm_e_M0)
        # change from baseline
        dPi_i_M0 = Pi_i_M0 - Pi_i_0
        dPi_e_M0 = Pi_e_M0 - Pi_e_0

        # osmotic pressures (kPa)
        dPi_m_M0 = (dPi_i_M0 - dPi_e_M0)/1000
        dPi_m_M1 = (dPi_i_M1 - dPi_e_M1)/1000
        dPi_m_M2 = (dPi_i_M2 - dPi_e_M2)/1000

        # water potentials (kPa)
        dPsi_M0 = dPi_e_M0/1000
        dPsi_M1 = (dp_e_M1+dPi_e_M1)/1000
        dPsi_M2 = (dp_e_M2+dPi_e_M2)/1000

        # project tp function space
        dOsm_e_M0 = self.project_to_function_space(dOsm_e_M0)
        dOsm_e_M1 = self.project_to_function_space(dOsm_e_M1)
        dOsm_e_M2 = self.project_to_function_space(dOsm_e_M2)
        dOsm_i_M0 = self.project_to_function_space(dOsm_i_M0)
        dOsm_i_M1 = self.project_to_function_space(dOsm_i_M1)
        dOsm_i_M2 = self.project_to_function_space(dOsm_i_M2)
        dPi_m_M0 = self.project_to_function_space(dPi_m_M0)
        dPi_m_M1 = self.project_to_function_space(dPi_m_M1)
        dPi_m_M2 = self.project_to_function_space(dPi_m_M2)
        dp_e_M1 = self.project_to_function_space(dp_e_M1/1000)  # convert to kPa
        dp_e_M2 = self.project_to_function_space(dp_e_M2/1000)
        dPsi_M0 = self.project_to_function_space(dPsi_M0)
        dPsi_M1 = self.project_to_function_space(dPsi_M1)
        dPsi_M2 = self.project_to_function_space(dPsi_M2)
        dPi_e_M0 = self.project_to_function_space(dPi_e_M0/1000)
        dPi_e_M1 = self.project_to_function_space(dPi_e_M1/1000)
        dPi_e_M2 = self.project_to_function_space(dPi_e_M2/1000)

        # create plot
        fig = plt.figure(figsize=(15*fs, 10*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(2, 3, 1, xlim=xlim, ylim=[-40, 15])
        plt.ylabel(r'$\Delta$O ECS (mM)', fontsize=fosi)
        plt.xticks(xticks, xticklabels)
        df.plot(dOsm_e_M0, color=colors[3], linewidth=lw, label='M0')
        df.plot(dOsm_e_M1, color=colors[0], linewidth=lw, label='M1')
        df.plot(dOsm_e_M2, color=colors[2], linewidth=lw, label='M2')

        ax2 = fig.add_subplot(2, 3, 2, xlim=xlim, ylim=[-40, 15])
        plt.ylabel(r'$\Delta$O ICS (mM)', fontsize=fosi)
        plt.xticks(xticks, xticklabels)
        df.plot(dOsm_i_M0, color=colors[3], linewidth=lw)
        df.plot(dOsm_i_M1, color=colors[0], linewidth=lw)
        df.plot(dOsm_i_M2, color=colors[2], linewidth=lw)

        ax3 = fig.add_subplot(2, 3, 3, xlim=xlim, ylim=[-130, 10])
        plt.ylabel(r'$\Delta (\Pi_\mathrm{i} - \Pi_\mathrm{e})$ (kPa)', fontsize=fosi)
        plt.xticks(xticks, xticklabels)
        df.plot(dPi_m_M0, color=colors[3], linewidth=lw)
        df.plot(dPi_m_M1, color=colors[0], linewidth=lw)
        df.plot(dPi_m_M2, color=colors[2], linewidth=lw)

        ax4 = fig.add_subplot(2, 3, 4, xlim=xlim, ylim=[-40, 100])
        plt.ylabel(r'$\Delta \Pi_\mathrm{e}$ (kPa)', fontsize=fosi)
        plt.xlabel(xlabel_x, fontsize=fosi)
        plt.xticks(xticks, xticklabels)
        df.plot(dPi_e_M0, color=colors[3], linewidth=lw)
        df.plot(dPi_e_M1, color=colors[0], linewidth=lw)
        df.plot(dPi_e_M2, color=colors[2], linewidth=lw)

        ax5 = fig.add_subplot(2, 3, 5, xlim=xlim, ylim=[-6, 1])
        plt.ylabel(r'$\Delta p_\mathrm{e}$ (kPa)', fontsize=fosi)
        plt.xlabel(xlabel_x, fontsize=fosi)
        plt.xticks(xticks, xticklabels)
        ax5.axhline(y=0, color=colors[3], linewidth=lw)
        df.plot(dp_e_M1, color=colors[0], linewidth=lw)
        df.plot(dp_e_M2, color=colors[2], linewidth=lw)

        ax6 = fig.add_subplot(2, 3, 6, xlim=xlim, ylim=[-40, 100])
        plt.ylabel(r'$\Delta (\Pi_\mathrm{e} + p_\mathrm{e})$ (kPa)', fontsize=fosi)
        plt.xlabel(xlabel_x, fontsize=fosi)
        plt.xticks(xticks, xticklabels)
        df.plot(dPsi_M0, color=colors[3], linewidth=lw)
        df.plot(dPsi_M1, color=colors[0], linewidth=lw)
        df.plot(dPsi_M2, color=colors[2], linewidth=lw)

        if self.verbose:
            print('max O_e decrease M0:', min(dOsm_e_M0.vector()[:]))
            print('max O_e decrease M1:', min(dOsm_e_M1.vector()[:]))
            print('max O_e decrease M2:', min(dOsm_e_M2.vector()[:]))
            print('max O_i increase M0:', max(dOsm_i_M0.vector()[:]))
            print('max O_i decrease M1:', min(dOsm_i_M1.vector()[:]))
            print('max O_i increase M2:', max(dOsm_i_M2.vector()[:]))
            print('max Pi_m decrease M0:', min(dPi_m_M0.vector()[:]))
            print('max Pi_m decrease M1:', min(dPi_m_M1.vector()[:]))
            print('max Pi_m decrease M2:', min(dPi_m_M2.vector()[:]))
            print('max Pi_e increase M0', max(dPi_e_M0.vector()[:]))
            print('max Pi_e increase M1', max(dPi_e_M1.vector()[:]))
            print('max Pi_e increase M2', max(dPi_e_M2.vector()[:]))
            print('max p_e decrease M1', min(dp_e_M1.vector()[:]))
            print('max p_e decrease M2', min(dp_e_M2.vector()[:]))
            print('max Psi increase M0', max(dPsi_M0.vector()[:]))
            print('max Psi increase M1', max(dPsi_M1.vector()[:]))
            print('max Psi increase M2', max(dPsi_M2.vector()[:]))

        plt.figlegend(bbox_to_anchor=(0.17, 0.68))

        # make pretty
        ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}', r'\textbf{E}', r'\textbf{F}']
        for num, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
            ax.text(-0.2, 1.02, letters[num], transform=ax.transAxes, size=22, weight='bold')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # save figure to file
        fname_res = path_figs + 'Figure4'
        plt.savefig(fname_res + '.pdf', format='pdf')
        plt.close()

        return

    def plot_M2_and_M3_fluid_velocities(self, path_figs, n):
        """ plot fluid velocities at t=n """

        # get parameters
        temperature = self.model.params['temperature']
        R = self.model.params['R']
        a_i = self.model.params['a'][0]
        kappa = self.model.params['kappa']
        eps_r = self.model.params['eps_r']
        eps_zero = self.model.params['eps_zero']
        zeta = self.model.params['zeta']
        mu = self.model.params['mu']
        K_m = self.model.params['K_m']
        p_m_init = self.model.params['p_m_init']

        # get data M2
        alpha_i = self.read_from_file_II(n, 0)
        phi_e = self.read_from_file_II(n, 8)
        p_e = self.read_from_file_II(n, 9)

        # extracellular volume fraction
        alpha_e = 0.6 - alpha_i

        # intracellular hydrostatic pressure
        tau = K_m*(alpha_i - float(self.model.alpha_i_init))
        p_i = p_e + tau + p_m_init

        # ICS fluid velocities
        u_i_hyd_ = - kappa[0]*df.grad(p_i)
        u_i_osm_ = kappa[0]*R*temperature*df.grad(a_i/alpha_i)
        u_i_tot_ = u_i_hyd_[0] + u_i_osm_[0]

        # ECS fluid velocities
        u_e_hyd_ = - kappa[1]*df.grad(p_e)
        u_e_tot_ = u_e_hyd_[0]

        # project to function space
        u_i_hyd = self.project_to_function_space(alpha_i*u_i_hyd_[0]*1.0e6*60)  # convert to um/min
        u_i_osm = self.project_to_function_space(alpha_i*u_i_osm_[0]*1.0e6*60)
        u_i_tot = self.project_to_function_space(alpha_i*u_i_tot_*1.0e6*60)
        u_e_hyd = self.project_to_function_space(alpha_e*u_e_hyd_[0]*1.0e6*60)
        u_e_tot = self.project_to_function_space(alpha_e*u_e_tot_*1.0e6*60)

        # create plot
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()

        ax1 = fig.add_subplot(2, 2, 1, xlim=xlim, ylim=[-90, 90])
        plt.title(r'(M2)', fontsize=fosi)
        plt.ylabel(r'$\alpha_\mathrm{i} u_\mathrm{i}$ ($\mu$m/min)', fontsize=fosi)
        plt.xticks(xticks, xticklabels)
        df.plot(u_i_osm, color=c5, linewidth=lw)
        df.plot(u_i_hyd, color=c3, linewidth=lw, linestyle='dotted')
        df.plot(u_i_tot, color='k', linewidth=lw, linestyle='dashed')

        ax2 = fig.add_subplot(2, 2, 2, xlim=xlim, ylim=[-90, 90])
        plt.title(r'(M2)', fontsize=fosi)
        plt.ylabel(r'$\alpha_\mathrm{e} u_\mathrm{e}$ ($\mu$m/min)', fontsize=fosi)
        plt.xticks(xticks, xticklabels)
        df.plot(u_e_hyd, color=c3, linewidth=lw, linestyle='dotted')
        df.plot(u_e_tot, color='k', linewidth=lw, linestyle='dashed')

        # get data M3
        alpha_i = self.read_from_file_III(n, 0)
        alpha_i = self.read_from_file_III(n, 0)
        phi_e = self.read_from_file_III(n, 8)
        p_e = self.read_from_file_III(n, 9)

        # extracellular volume fraction
        alpha_e = 0.6 - alpha_i

        # intracellular hydrostatic pressure
        tau = K_m*(alpha_i - float(self.model.alpha_i_init))
        p_i = p_e + tau + p_m_init

        # ICS fluid velocities
        u_i_hyd_ = - kappa[0]*df.grad(p_i)
        u_i_osm_ = kappa[0]*R*temperature*df.grad(a_i/alpha_i)
        u_i_tot_ = u_i_hyd_[0] + u_i_osm_[0]

        # ECS fluid velocities
        u_e_hyd_ = - kappa[1]*df.grad(p_e)
        u_e_eof_ = - eps_r*eps_zero*zeta*df.grad(phi_e)/mu
        u_e_tot_ = u_e_hyd_[0] + u_e_eof_[0]

        # project to function space
        u_i_hyd = self.project_to_function_space(alpha_i*u_i_hyd_[0]*1.0e6*60)  # convert to um/min
        u_i_osm = self.project_to_function_space(alpha_i*u_i_osm_[0]*1.0e6*60)
        u_i_tot = self.project_to_function_space(alpha_i*u_i_tot_*1.0e6*60)
        u_e_hyd = self.project_to_function_space(alpha_e*u_e_hyd_[0]*1.0e6*60)
        u_e_eof = self.project_to_function_space(alpha_e*u_e_eof_[0]*1.0e6*60)
        u_e_tot = self.project_to_function_space(alpha_e*u_e_tot_*1.0e6*60)

        ax3 = fig.add_subplot(2, 2, 3, xlim=xlim, ylim=[-90, 90])
        plt.title(r'(M3)', fontsize=fosi)
        plt.ylabel(r'$\alpha_\mathrm{i} u_\mathrm{i}$ ($\mu$m/min)', fontsize=fosi)
        plt.xlabel(xlabel_x, fontsize=fosi)
        plt.xticks(xticks, xticklabels)
        df.plot(u_i_osm, color=c5, label='osmotic', linewidth=lw)
        df.plot(u_i_hyd, color=c3, linewidth=lw, linestyle='dotted')
        df.plot(u_i_tot, color='k', linewidth=lw, linestyle='dashed')

        ax4 = fig.add_subplot(2, 2, 4, xlim=xlim, ylim=[-90, 90])
        plt.title(r'(M3)', fontsize=fosi)
        plt.ylabel(r'$\alpha_\mathrm{e} u_\mathrm{e}$ ($\mu$m/min)', fontsize=fosi)
        plt.xlabel(xlabel_x, fontsize=fosi)
        plt.xticks(xticks, xticklabels)
        df.plot(u_e_hyd, label='hydrostatic', color=c3, linewidth=lw, linestyle='dotted')
        df.plot(u_e_eof, label='electro-osmotic', color=c4, linewidth=lw)
        df.plot(u_e_tot, label='total', color='k', linewidth=lw, linestyle='dashed')

        plt.figlegend(bbox_to_anchor=(0.98, 0.94))

        # make pretty
        ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}']
        for num, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(-0.2, 1.0, letters[num], transform=ax.transAxes, size=22, weight='bold')
            # make pretty
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # save figure to file
        fname_res = path_figs + 'Figure5'
        plt.savefig(fname_res + '.pdf', format='pdf')
        plt.close()

        return

    def plot_ion_fluxes(self, path_figs, n):

        # get parameters
        z = self.model.params['z']
        temperature = self.model.params['temperature']
        R = self.model.params['R']
        F = self.model.params['F']
        a_i_ = self.model.params['a'][0]
        z = self.model.params['z']
        z_Na = z[0]
        z_K = z[1]
        z_Cl = z[2]
        kappa = self.model.params['kappa']
        eps_r = self.model.params['eps_r']
        eps_zero = self.model.params['eps_zero']
        zeta = self.model.params['zeta']
        mu = self.model.params['mu']
        K_m = self.model.params['K_m']
        p_m_init = self.model.params['p_m_init']
        alpha_i_init = float(self.model.alpha_i_init)
        lambda_i = self.model.params['lambdas'][0]
        lambda_e = self.model.params['lambdas'][1]
        D = self.model.params['D']

        # get data M1
        alpha_i_ = self.read_from_file_I(n, 0)
        Na_i_ = self.read_from_file_I(n, 1)
        Na_e_ = self.read_from_file_I(n, 2)
        K_i_ = self.read_from_file_I(n, 3)
        K_e_ = self.read_from_file_I(n, 4)
        Cl_i_ = self.read_from_file_I(n, 5)
        Cl_e_ = self.read_from_file_I(n, 6)
        phi_i_ = self.read_from_file_I(n, 7)
        phi_e_ = self.read_from_file_I(n, 8)
        p_e_ = self.read_from_file_I(n, 9)

        # extracellular volume fraction
        alpha_e_ = 0.6 - alpha_i_

        # intracellular hydrostatic pressure
        tau = K_m*(alpha_i_ - alpha_i_init)
        p_i_ = p_e_ + tau + p_m_init

        # intra- and extracellular fluid velocities
        u_i_ = - kappa[0]*df.grad(p_i_)
        u_e_ = - kappa[1]*df.grad(p_e_)

        # axial ion fluxes
        j_e_Na_ = - (D[0]/lambda_e**2)*(df.grad(Na_e_) + z_Na*F*Na_e_/(R*temperature)*df.grad(phi_e_)) + u_e_*Na_e_
        j_e_K_ = - (D[1]/lambda_e**2)*(df.grad(K_e_) + z_K*F*K_e_/(R*temperature)*df.grad(phi_e_)) + u_e_*K_e_
        j_e_Cl_ = - (D[2]/lambda_e**2)*(df.grad(Cl_e_) + z_Cl*F*Cl_e_/(R*temperature)*df.grad(phi_e_)) + u_e_*Cl_e_
        j_i_Na_ = - (D[0]/lambda_i**2)*(df.grad(Na_i_) + z_Na*F*Na_i_/(R*temperature)*df.grad(phi_i_)) + u_i_*Na_i_
        j_i_K_ = - (D[1]/lambda_i**2)*(df.grad(K_i_) + z_K*F*K_i_/(R*temperature)*df.grad(phi_i_)) + u_i_*K_i_
        j_i_Cl_ = - (D[2]/lambda_i**2)*(df.grad(Cl_i_) + z_Cl*F*Cl_i_/(R*temperature)*df.grad(phi_i_)) + u_i_*Cl_i_

        j_i_diff_K_ = - (D[1]/lambda_i**2)*df.grad(K_i_)
        j_i_field_K_ = - (D[1]/lambda_i**2)*(z_K*F*K_i_/(R*temperature)*df.grad(phi_i_))
        j_i_flow_K_ = u_i_*K_i_

        j_e_diff_K_ = - (D[1]/lambda_e**2)*df.grad(K_e_)
        j_e_field_K_ = - (D[1]/lambda_e**2)*(z_K*F*K_e_/(R*temperature)*df.grad(phi_e_))
        j_e_flow_K_ = u_e_*K_e_

        j_i_diff_Na_ = - (D[0]/lambda_i**2)*df.grad(Na_i_)
        j_i_field_Na_ = - (D[0]/lambda_i**2)*(z_Na*F*Na_i_/(R*temperature)*df.grad(phi_i_))
        j_i_flow_Na_ = u_i_*Na_i_

        j_e_diff_Na_ = - (D[0]/lambda_e**2)*df.grad(Na_e_)
        j_e_field_Na_ = - (D[0]/lambda_e**2)*(z_Na*F*Na_e_/(R*temperature)*df.grad(phi_e_))
        j_e_flow_Na_ = u_e_*Na_e_

        j_i_diff_Cl_ = - (D[2]/lambda_i**2)*df.grad(Cl_i_)
        j_i_field_Cl_ = - (D[2]/lambda_i**2)*(z_Cl*F*Cl_i_/(R*temperature)*df.grad(phi_i_))
        j_i_flow_Cl_ = u_i_*Cl_i_

        j_e_diff_Cl_ = - (D[2]/lambda_e**2)*df.grad(Cl_e_)
        j_e_field_Cl_ = - (D[2]/lambda_e**2)*(z_Cl*F*Cl_e_/(R*temperature)*df.grad(phi_e_))
        j_e_flow_Cl_ = u_e_*Cl_e_

        j_i_q_ = j_i_K_ + j_i_Na_ - j_i_Cl_
        j_e_q_ = j_e_K_ + j_e_Na_ - j_e_Cl_

        # project to function space
        j_i_q = self.project_to_function_space(alpha_i_*j_i_q_[0]*1e6)
        j_e_q = self.project_to_function_space(alpha_e_*j_e_q_[0]*1e6)
        j_i_Na = self.project_to_function_space(alpha_i_*j_i_Na_[0]*1e6)
        j_i_K = self.project_to_function_space(alpha_i_*j_i_K_[0]*1e6)
        j_i_Cl = self.project_to_function_space(alpha_i_*j_i_Cl_[0]*1e6)
        j_e_Na = self.project_to_function_space(alpha_e_*j_e_Na_[0]*1e6)
        j_e_K = self.project_to_function_space(alpha_e_*j_e_K_[0]*1e6)
        j_e_Cl = self.project_to_function_space(alpha_e_*j_e_Cl_[0]*1e6)
        j_i_diff_K = self.project_to_function_space(alpha_i_*j_i_diff_K_[0]*1e6)
        j_i_field_K = self.project_to_function_space(alpha_i_*j_i_field_K_[0]*1e6)
        j_i_flow_K = self.project_to_function_space(alpha_i_*j_i_flow_K_[0]*1e6)
        j_e_diff_K = self.project_to_function_space(alpha_e_*j_e_diff_K_[0]*1e6)
        j_e_field_K = self.project_to_function_space(alpha_e_*j_e_field_K_[0]*1e6)
        j_e_flow_K = self.project_to_function_space(alpha_e_*j_e_flow_K_[0]*1e6)
        j_i_diff_Na = self.project_to_function_space(alpha_i_*j_i_diff_Na_[0]*1e6)
        j_i_field_Na = self.project_to_function_space(alpha_i_*j_i_field_Na_[0]*1e6)
        j_i_flow_Na = self.project_to_function_space(alpha_i_*j_i_flow_Na_[0]*1e6)
        j_e_diff_Na = self.project_to_function_space(alpha_e_*j_e_diff_Na_[0]*1e6)
        j_e_field_Na = self.project_to_function_space(alpha_e_*j_e_field_Na_[0]*1e6)
        j_e_flow_Na = self.project_to_function_space(alpha_e_*j_e_flow_Na_[0]*1e6)
        j_i_diff_Cl = self.project_to_function_space(alpha_i_*j_i_diff_Cl_[0]*1e6)
        j_i_field_Cl = self.project_to_function_space(alpha_i_*j_i_field_Cl_[0]*1e6)
        j_i_flow_Cl = self.project_to_function_space(alpha_i_*j_i_flow_Cl_[0]*1e6)
        j_e_diff_Cl = self.project_to_function_space(alpha_e_*j_e_diff_Cl_[0]*1e6)
        j_e_field_Cl = self.project_to_function_space(alpha_e_*j_e_field_Cl_[0]*1e6)
        j_e_flow_Cl = self.project_to_function_space(alpha_e_*j_e_flow_Cl_[0]*1e6)

        if self.verbose:
            print('Max j_i_K M1:', j_i_K.vector().max())
            print('Max j_i_q M1:', j_i_q.vector().max())
            print('Max j_e_q M1:', j_e_q.vector().max())

        # advection/diffusion- and advection/drift fractions
        max_index_j_i_K = np.argmax(j_i_K.vector()[:])
        F_K_i_diff = j_i_flow_K.vector()[max_index_j_i_K] / (j_i_diff_K.vector()[max_index_j_i_K])
        F_K_i_drift = j_i_flow_K.vector()[max_index_j_i_K] / (j_i_field_K.vector()[max_index_j_i_K])

        max_index_j_e_K = np.argmax(j_e_K.vector()[:])
        F_K_e_diff = j_e_flow_K.vector()[max_index_j_e_K] / (j_e_diff_K.vector()[max_index_j_e_K])
        F_K_e_drift = j_e_flow_K.vector()[max_index_j_e_K] / (j_e_field_K.vector()[max_index_j_e_K])

        max_index_j_i_Na = np.argmax(j_i_Na.vector()[:])
        F_Na_i_diff = j_i_flow_Na.vector()[max_index_j_i_Na] / (j_i_diff_Na.vector()[max_index_j_i_Na])
        F_Na_i_drift = j_i_flow_Na.vector()[max_index_j_i_Na] / (j_i_field_Na.vector()[max_index_j_i_Na])

        max_index_j_e_Na = np.argmax(j_e_Na.vector()[:])
        F_Na_e_diff = j_e_flow_Na.vector()[max_index_j_e_Na] / (j_e_diff_Na.vector()[max_index_j_e_Na])
        F_Na_e_drift = j_e_flow_Na.vector()[max_index_j_e_Na] / (j_e_diff_Na.vector()[max_index_j_e_Na] + j_e_field_Na.vector()[max_index_j_e_Na])

        max_index_j_i_Cl = np.argmax(j_i_Cl.vector()[:])
        F_Cl_i_diff = j_i_flow_Cl.vector()[max_index_j_i_Cl] / (j_i_diff_Cl.vector()[max_index_j_i_Cl])
        F_Cl_i_drift = j_i_flow_Cl.vector()[max_index_j_i_Cl] / (j_i_field_Cl.vector()[max_index_j_i_Cl])

        max_index_j_e_Cl = np.argmax(j_e_Cl.vector()[:])
        F_Cl_e_diff = j_e_flow_Cl.vector()[max_index_j_e_Cl] / (j_e_diff_Cl.vector()[max_index_j_e_Cl])
        F_Cl_e_drift = j_e_flow_Cl.vector()[max_index_j_e_Cl] / (j_e_field_Cl.vector()[max_index_j_e_Cl])

        # create plot
        fig = plt.figure(figsize=(15*fs, 20*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(4, 3, 1, xlim=xlim, ylim=[-30, 30])
        plt.xticks(xticks, xticklabels)
        plt.title('ECS K$^+$ flux (M1)')
        plt.ylabel(r'$\alpha_\mathrm{e}j_\mathrm{e}$ ($\mu$mol/(m$^2$s)')
        df.plot(j_e_diff_K, label='diffusion', color=c3, linewidth=3)
        df.plot(j_e_field_K, label='electric drift', color=c4, linewidth=3)
        df.plot(j_e_flow_K, label='advection', color=c5, linewidth=3)
        df.plot(j_e_K, color='dimgrey', linewidth=3, linestyle='dashed')
        ax1.text(0.03, 0.13, r'F\textsubscript{diff} = ' + str(abs(round(F_K_e_diff, 3))), horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, size=16)
        ax1.text(0.03, 0.05, r'F\textsubscript{drift} = ' + str(abs(round(F_K_e_drift, 3))), horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, size=16)

        ax2 = fig.add_subplot(4, 3, 2, xlim=xlim, ylim=[-80, 80])
        plt.xticks(xticks, xticklabels)
        plt.title('ECS Na$^+$ flux (M1)')
        df.plot(j_e_diff_Na, color=c3, linewidth=3)
        df.plot(j_e_field_Na, color=c4, linewidth=3)
        df.plot(j_e_flow_Na, color=c5, linewidth=3)
        df.plot(j_e_Na, color='dimgrey', linewidth=3, linestyle='dashed')
        ax2.text(0.03, 0.13, r'F\textsubscript{diff} = ' + str(abs(round(F_Na_e_diff, 3))), horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes, size=16)
        ax2.text(0.03, 0.05, r'F\textsubscript{drift} = ' + str(abs(round(F_Na_e_drift, 3))), horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes, size=16)

        ax3 = fig.add_subplot(4, 3, 3, xlim=xlim, ylim=[-80, 80])
        plt.yticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])
        plt.xticks(xticks, xticklabels)
        plt.title('ECS Cl$^-$ flux (M1)')
        df.plot(j_e_diff_Cl, color=c3, linewidth=3)
        df.plot(j_e_field_Cl, color=c4, linewidth=3)
        df.plot(j_e_flow_Cl, color=c5, linewidth=3)
        df.plot(j_e_Cl, color='dimgrey', linewidth=3, linestyle='dashed')
        ax3.text(0.03, 0.13, r'F\textsubscript{diff} = ' + str(abs(round(F_Cl_e_diff, 3))), horizontalalignment='left', verticalalignment='center', transform=ax3.transAxes, size=16)
        ax3.text(0.03, 0.05, r'F\textsubscript{drift} = ' + str(abs(round(F_Cl_e_drift, 3))), horizontalalignment='left', verticalalignment='center', transform=ax3.transAxes, size=16)

        ax4 = fig.add_subplot(4, 3, 4, xlim=xlim, ylim=[-80, 80])
        plt.xticks(xticks, xticklabels)
        plt.title('ICS K$^+$ flux (M1)')
        plt.ylabel(r'$\alpha_\mathrm{i}j_\mathrm{i}$ ($\mu$mol/(m$^2$s)')
        df.plot(j_i_diff_K, color=c3, linewidth=3)
        df.plot(j_i_field_K, color=c4, linewidth=3)
        df.plot(j_i_flow_K, color=c5, linewidth=3)
        df.plot(j_i_K, color='dimgrey', label='total', linewidth=3, linestyle='dashed')
        ax4.text(0.03, 0.13, r'F\textsubscript{diff} = ' + str(abs(round(F_K_i_diff, 3))), horizontalalignment='left', verticalalignment='center', transform=ax4.transAxes, size=16)
        ax4.text(0.03, 0.05, r'F\textsubscript{drift} = ' + str(abs(round(F_K_i_drift, 3))), horizontalalignment='left', verticalalignment='center', transform=ax4.transAxes, size=16)

        ax5 = fig.add_subplot(4, 3, 5, xlim=xlim, ylim=[-10, 10])
        plt.xticks(xticks, xticklabels)
        plt.title('ICS Na$^+$ (M1)')
        df.plot(j_i_diff_Na, color=c3, linewidth=3)
        df.plot(j_i_field_Na, color=c4, linewidth=3)
        df.plot(j_i_flow_Na, color=c5, linewidth=3)
        df.plot(j_i_Na, color='dimgrey', linewidth=3, linestyle='dashed')
        ax5.text(0.03, 0.13, r'F\textsubscript{diff} = ' + str(abs(round(F_Na_i_diff, 3))), horizontalalignment='left', verticalalignment='center', transform=ax5.transAxes, size=16)
        ax5.text(0.03, 0.05, r'F\textsubscript{drift} = ' + str(abs(round(F_Na_i_drift, 3))), horizontalalignment='left', verticalalignment='center', transform=ax5.transAxes, size=16)

        ax6 = fig.add_subplot(4, 3, 6, xlim=xlim, ylim=[-10, 10])
        plt.xticks(xticks, xticklabels)
        plt.title('ICS Cl$^-$ (M1)')
        df.plot(j_i_diff_Cl, color=c3, linewidth=3)
        df.plot(j_i_field_Cl, color=c4, linewidth=3)
        df.plot(j_i_flow_Cl, color=c5, linewidth=3)
        df.plot(j_i_Cl, color='dimgrey', linewidth=3, linestyle='dashed')
        ax6.text(0.03, 0.13, r'F\textsubscript{diff} = ' + str(abs(round(F_Cl_i_diff, 3))), horizontalalignment='left', verticalalignment='center', transform=ax6.transAxes, size=16)
        ax6.text(0.03, 0.05, r'F\textsubscript{drift} = ' + str(abs(round(F_Cl_i_drift, 3))), horizontalalignment='left', verticalalignment='center', transform=ax6.transAxes, size=16)

        # get data M3
        alpha_i_ = self.read_from_file_III(n, 0)
        Na_i_ = self.read_from_file_III(n, 1)
        Na_e_ = self.read_from_file_III(n, 2)
        K_i_ = self.read_from_file_III(n, 3)
        K_e_ = self.read_from_file_III(n, 4)
        Cl_i_ = self.read_from_file_III(n, 5)
        Cl_e_ = self.read_from_file_III(n, 6)
        phi_i_ = self.read_from_file_III(n, 7)
        phi_e_ = self.read_from_file_III(n, 8)
        p_e_ = self.read_from_file_III(n, 9)

        # extracellular volume fraction
        alpha_e_ = 0.6 - alpha_i_

        # intracellular hydrostatic pressure
        tau = K_m*(alpha_i_ - alpha_i_init)
        p_i_ = p_e_ + tau + p_m_init

        # intra- and extracellular fluid velocities
        u_i_ = - kappa[0]*(df.grad(p_i_) - R*temperature*df.grad(a_i_/alpha_i_))
        u_e_ = - kappa[1]*df.grad(p_e_) - eps_r*eps_zero*zeta*df.grad(phi_e_)/mu

        # axial ion fluxes
        j_e_Na_ = - (D[0]/lambda_e**2)*(df.grad(Na_e_)
                   + z_Na*F*Na_e_/(R*temperature)*df.grad(phi_e_)) + u_e_*Na_e_

        j_e_K_ = - (D[1]/lambda_e**2)*(df.grad(K_e_)
                   + z_K*F*K_e_/(R*temperature)*df.grad(phi_e_)) + u_e_*K_e_

        j_e_Cl_ = - (D[2]/lambda_e**2)*(df.grad(Cl_e_)
                   + z_Cl*F*Cl_e_/(R*temperature)*df.grad(phi_e_)) + u_e_*Cl_e_

        j_i_Na_ = - (D[0]/lambda_i**2)*(df.grad(Na_i_)
                   + z_Na*F*Na_i_/(R*temperature)*df.grad(phi_i_)) + u_i_*Na_i_

        j_i_K_ = - (D[1]/lambda_i**2)*(df.grad(K_i_)
                   + z_K*F*K_i_/(R*temperature)*df.grad(phi_i_)) + u_i_*K_i_

        j_i_Cl_ = - (D[2]/lambda_i**2)*(df.grad(Cl_i_)
                   + z_Cl*F*Cl_i_/(R*temperature)*df.grad(phi_i_)) + u_i_*Cl_i_

        j_i_diff_K_ = - (D[1]/lambda_i**2)*df.grad(K_i_)
        j_i_field_K_ = - (D[1]/lambda_i**2)*(z_K*F*K_i_/(R*temperature)*df.grad(phi_i_))
        j_i_flow_K_ = u_i_*K_i_

        j_e_diff_K_ = - (D[1]/lambda_e**2)*df.grad(K_e_)
        j_e_field_K_ = - (D[1]/lambda_e**2)*(z_K*F*K_e_/(R*temperature)*df.grad(phi_e_))
        j_e_flow_K_ = u_e_*K_e_

        j_i_diff_Na_ = - (D[0]/lambda_i**2)*df.grad(Na_i_)
        j_i_field_Na_ = - (D[0]/lambda_i**2)*(z_Na*F*Na_i_/(R*temperature)*df.grad(phi_i_))
        j_i_flow_Na_ = u_i_*Na_i_

        j_e_diff_Na_ = - (D[0]/lambda_e**2)*df.grad(Na_e_)
        j_e_field_Na_ = - (D[0]/lambda_e**2)*(z_Na*F*Na_e_/(R*temperature)*df.grad(phi_e_))
        j_e_flow_Na_ = u_e_*Na_e_

        j_i_diff_Cl_ = - (D[2]/lambda_i**2)*df.grad(Cl_i_)
        j_i_field_Cl_ = - (D[2]/lambda_i**2)*(z_Cl*F*Cl_i_/(R*temperature)*df.grad(phi_i_))
        j_i_flow_Cl_ = u_i_*Cl_i_

        j_e_diff_Cl_ = - (D[2]/lambda_e**2)*df.grad(Cl_e_)
        j_e_field_Cl_ = - (D[2]/lambda_e**2)*(z_Cl*F*Cl_e_/(R*temperature)*df.grad(phi_e_))
        j_e_flow_Cl_ = u_e_*Cl_e_

        j_i_q_ = j_i_K_ + j_i_Na_ - j_i_Cl_
        j_e_q_ = j_e_K_ + j_e_Na_ - j_e_Cl_

        # project to function space
        j_i_q = self.project_to_function_space(alpha_i_*j_i_q_[0]*1e6)
        j_e_q = self.project_to_function_space(alpha_e_*j_e_q_[0]*1e6)
        j_i_Na = self.project_to_function_space(alpha_i_*j_i_Na_[0]*1e6)
        j_i_K = self.project_to_function_space(alpha_i_*j_i_K_[0]*1e6)
        j_i_Cl = self.project_to_function_space(alpha_i_*j_i_Cl_[0]*1e6)
        j_e_Na = self.project_to_function_space(alpha_e_*j_e_Na_[0]*1e6)
        j_e_K = self.project_to_function_space(alpha_e_*j_e_K_[0]*1e6)
        j_e_Cl = self.project_to_function_space(alpha_e_*j_e_Cl_[0]*1e6)
        j_i_diff_K = self.project_to_function_space(alpha_i_*j_i_diff_K_[0]*1e6)
        j_i_field_K = self.project_to_function_space(alpha_i_*j_i_field_K_[0]*1e6)
        j_i_flow_K = self.project_to_function_space(alpha_i_*j_i_flow_K_[0]*1e6)
        j_e_diff_K = self.project_to_function_space(alpha_e_*j_e_diff_K_[0]*1e6)
        j_e_field_K = self.project_to_function_space(alpha_e_*j_e_field_K_[0]*1e6)
        j_e_flow_K = self.project_to_function_space(alpha_e_*j_e_flow_K_[0]*1e6)
        j_i_diff_Na = self.project_to_function_space(alpha_i_*j_i_diff_Na_[0]*1e6)
        j_i_field_Na = self.project_to_function_space(alpha_i_*j_i_field_Na_[0]*1e6)
        j_i_flow_Na = self.project_to_function_space(alpha_i_*j_i_flow_Na_[0]*1e6)
        j_e_diff_Na = self.project_to_function_space(alpha_e_*j_e_diff_Na_[0]*1e6)
        j_e_field_Na = self.project_to_function_space(alpha_e_*j_e_field_Na_[0]*1e6)
        j_e_flow_Na = self.project_to_function_space(alpha_e_*j_e_flow_Na_[0]*1e6)
        j_i_diff_Cl = self.project_to_function_space(alpha_i_*j_i_diff_Cl_[0]*1e6)
        j_i_field_Cl = self.project_to_function_space(alpha_i_*j_i_field_Cl_[0]*1e6)
        j_i_flow_Cl = self.project_to_function_space(alpha_i_*j_i_flow_Cl_[0]*1e6)
        j_e_diff_Cl = self.project_to_function_space(alpha_e_*j_e_diff_Cl_[0]*1e6)
        j_e_field_Cl = self.project_to_function_space(alpha_e_*j_e_field_Cl_[0]*1e6)
        j_e_flow_Cl = self.project_to_function_space(alpha_e_*j_e_flow_Cl_[0]*1e6)

        if self.verbose:
            print('Max j_i_K M3:', j_i_K.vector().max())
            print('Max j_i_q M3:', j_i_q.vector().max())
            print('Max j_e_q M3:', j_e_q.vector().max())

        # advection/diffusion- and advection/drift fractions
        max_index_j_i_K = np.argmax(j_i_K.vector()[:])
        F_K_i_diff = j_i_flow_K.vector()[max_index_j_i_K] / (j_i_diff_K.vector()[max_index_j_i_K])
        F_K_i_drift = j_i_flow_K.vector()[max_index_j_i_K] / (j_i_field_K.vector()[max_index_j_i_K])

        max_index_j_e_K = np.argmax(j_e_K.vector()[:])
        F_K_e_diff = j_e_flow_K.vector()[max_index_j_e_K] / (j_e_diff_K.vector()[max_index_j_e_K])
        F_K_e_drift = j_e_flow_K.vector()[max_index_j_e_K] / (j_e_field_K.vector()[max_index_j_e_K])

        max_index_j_i_Na = np.argmax(j_i_Na.vector()[:])
        F_Na_i_diff = j_i_flow_Na.vector()[max_index_j_i_Na] / (j_i_diff_Na.vector()[max_index_j_i_Na])
        F_Na_i_drift = j_i_flow_Na.vector()[max_index_j_i_Na] / (j_i_field_Na.vector()[max_index_j_i_Na])

        max_index_j_e_Na = np.argmax(j_e_Na.vector()[:])
        F_Na_e_diff = j_e_flow_Na.vector()[max_index_j_e_Na] / (j_e_diff_Na.vector()[max_index_j_e_Na])
        F_Na_e_drift = j_e_flow_Na.vector()[max_index_j_e_Na] / (j_e_diff_Na.vector()[max_index_j_e_Na] + j_e_field_Na.vector()[max_index_j_e_Na])

        max_index_j_i_Cl = np.argmax(j_i_Cl.vector()[:])
        F_Cl_i_diff = j_i_flow_Cl.vector()[max_index_j_i_Cl] / (j_i_diff_Cl.vector()[max_index_j_i_Cl])
        F_Cl_i_drift = j_i_flow_Cl.vector()[max_index_j_i_Cl] / (j_i_field_Cl.vector()[max_index_j_i_Cl])

        max_index_j_e_Cl = np.argmax(j_e_Cl.vector()[:])
        F_Cl_e_diff = j_e_flow_Cl.vector()[max_index_j_e_Cl] / (j_e_diff_Cl.vector()[max_index_j_e_Cl])
        F_Cl_e_drift = j_e_flow_Cl.vector()[max_index_j_e_Cl] / (j_e_field_Cl.vector()[max_index_j_e_Cl])

        ax7 = fig.add_subplot(4, 3, 7, xlim=xlim, ylim=[-30, 30])
        plt.xticks(xticks, xticklabels)
        plt.title('ECS K$^+$ flux (M3)')
        plt.ylabel(r'$\alpha_\mathrm{e}j_\mathrm{e}$ ($\mu$mol/(m$^2$s)')
        df.plot(j_e_diff_K, color=c3, linewidth=3)
        df.plot(j_e_field_K, color=c4, linewidth=3)
        df.plot(j_e_flow_K, color=c5, linewidth=3)
        df.plot(j_e_K, color='dimgrey', linewidth=3, linestyle='dashed')
        ax7.text(0.03, 0.13, r'F\textsubscript{diff} = ' + str(abs(round(F_K_e_diff, 3))),
                 horizontalalignment='left', verticalalignment='center', transform=ax7.transAxes, size=16)
        ax7.text(0.03, 0.05, r'F\textsubscript{drift} = ' + str(abs(round(F_K_e_drift, 3))),
                 horizontalalignment='left', verticalalignment='center', transform=ax7.transAxes, size=16)

        ax8 = fig.add_subplot(4, 3, 8, xlim=xlim, ylim=[-90, 90])
        plt.xticks(xticks, xticklabels)
        plt.title('ECS Na$^+$ flux (M3)')
        df.plot(j_e_diff_Na, color=c3, linewidth=3)
        df.plot(j_e_field_Na, color=c4, linewidth=3)
        df.plot(j_e_flow_Na, color=c5, linewidth=3)
        df.plot(j_e_Na, color='dimgrey', linewidth=3, linestyle='dashed')
        ax8.text(0.03, 0.13, r'F\textsubscript{diff} = ' + str(abs(round(F_Na_e_diff, 3))),
                 horizontalalignment='left', verticalalignment='center', transform=ax8.transAxes, size=16)
        ax8.text(0.03, 0.05, r'F\textsubscript{drift} = ' + str(abs(round(F_Na_e_drift, 3))),
                 horizontalalignment='left', verticalalignment='center', transform=ax8.transAxes, size=16)

        ax9 = fig.add_subplot(4, 3, 9, xlim=xlim, ylim=[-80, 80])
        plt.yticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])
        plt.xticks(xticks, xticklabels)
        plt.title('ECS Cl$^-$ flux (M3)')
        df.plot(j_e_diff_Cl, color=c3, linewidth=3)
        df.plot(j_e_field_Cl, color=c4, linewidth=3)
        df.plot(j_e_flow_Cl, color=c5, linewidth=3)
        df.plot(j_e_Cl, color='dimgrey', linewidth=3, linestyle='dashed')
        ax9.text(0.03, 0.13, r'F\textsubscript{diff} = ' + str(abs(round(F_Cl_e_diff, 3))),
                 horizontalalignment='left', verticalalignment='center', transform=ax9.transAxes, size=16)
        ax9.text(0.03, 0.05, r'F\textsubscript{drift} = ' + str(abs(round(F_Cl_e_drift, 3))),
                 horizontalalignment='left', verticalalignment='center', transform=ax9.transAxes, size=16)

        ax10 = fig.add_subplot(4, 3, 10, xlim=xlim, ylim=[-80, 80])
        plt.xticks(xticks, xticklabels)
        plt.xlabel(xlabel_x, fontsize=fosi)
        plt.title('ICS K$^+$ flux (M3)')
        plt.ylabel(r'$\alpha_\mathrm{i}j_\mathrm{i}$ ($\mu$mol/(m$^2$s)')
        df.plot(j_i_diff_K, color=c3, linewidth=3)
        df.plot(j_i_field_K, color=c4, linewidth=3)
        df.plot(j_i_flow_K, color=c5, linewidth=3)
        df.plot(j_i_K, color='dimgrey', linewidth=3, linestyle='dashed')
        ax10.text(0.03, 0.13, r'F\textsubscript{diff} = ' + str(abs(round(F_K_i_diff, 3))),
                  horizontalalignment='left', verticalalignment='center', transform=ax10.transAxes, size=16)
        ax10.text(0.03, 0.05, r'F\textsubscript{drift} = ' + str(abs(round(F_K_i_drift, 3))),
                  horizontalalignment='left', verticalalignment='center', transform=ax10.transAxes, size=16)

        ax11 = fig.add_subplot(4, 3, 11, xlim=xlim, ylim=[-10, 10])
        plt.xticks(xticks, xticklabels)
        plt.xlabel(xlabel_x, fontsize=fosi)
        plt.title('ICS Na$^+$ (M3)')
        df.plot(j_i_diff_Na, color=c3, linewidth=3)
        df.plot(j_i_field_Na, color=c4, linewidth=3)
        df.plot(j_i_flow_Na, color=c5, linewidth=3)
        df.plot(j_i_Na, color='dimgrey', linewidth=3, linestyle='dashed')
        ax11.text(0.03, 0.13, r'F\textsubscript{diff} = ' + str(abs(round(F_Na_i_diff, 3))),
                  horizontalalignment='left', verticalalignment='center', transform=ax11.transAxes, size=16)
        ax11.text(0.03, 0.05, r'F\textsubscript{drift} = ' + str(abs(round(F_Na_i_drift, 3))),
                  horizontalalignment='left', verticalalignment='center', transform=ax11.transAxes, size=16)

        ax12 = fig.add_subplot(4, 3, 12, xlim=xlim, ylim=[-10, 10])
        plt.xticks(xticks, xticklabels)
        plt.xlabel(xlabel_x, fontsize=fosi)
        plt.title('ICS Cl$^-$ (M3)')
        df.plot(j_i_diff_Cl, color=c3, linewidth=3)
        df.plot(j_i_field_Cl, color=c4, linewidth=3)
        df.plot(j_i_flow_Cl, color=c5, linewidth=3)
        df.plot(j_i_Cl, color='dimgrey', linewidth=3, linestyle='dashed')
        ax12.text(0.03, 0.13, r'F\textsubscript{diff} = ' + str(abs(round(F_Cl_i_diff, 3))),
                  horizontalalignment='left', verticalalignment='center', transform=ax12.transAxes, size=16)
        ax12.text(0.03, 0.05, r'F\textsubscript{drift} = ' + str(abs(round(F_Cl_i_drift, 3))),
                  horizontalalignment='left', verticalalignment='center', transform=ax12.transAxes, size=16)

        plt.figlegend(bbox_to_anchor=(1.0, 0.97))

        # make pretty
        ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(right=0.94)

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}',
                   r'\textbf{D}', r'\textbf{E}', r'\textbf{F}',
                   r'\textbf{G}', r'\textbf{H}', r'\textbf{I}',
                   r'\textbf{J}', r'\textbf{K}', r'\textbf{L}']
        for num, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]):
            ax.text(-0.2, 1.0, letters[num], transform=ax.transAxes, size=22, weight='bold')
            # make pretty
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # save figure to file
        fname_res = path_figs + 'Figure6'
        plt.savefig(fname_res + '.pdf', format='pdf')
        plt.close()

        return
