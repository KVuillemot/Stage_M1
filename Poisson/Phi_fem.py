# import des différents packages 
import dolfin as df 
import multiphenics as mph 
import matplotlib.pyplot as plt 
import numpy as np 
import mshr 
from vedo.dolfin import plot, screenshot, interactive, closePlotter, clear 
from matplotlib import rc, rcParams 

# plot parameters
plt.style.use('bmh') 
params = {'axes.labelsize': 26,
          'font.size': 22,
          'axes.titlesize': 'large',
          'legend.fontsize': 18,
          'figure.titlesize': 24,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'text.usetex': True,
          'figure.figsize':(10,8),          
          'legend.shadow': True,
          'patch.edgecolor': 'black'}
plt.rcParams.update(params)
# dolfin parameters
df.parameters["ghost_mode"] = "shared_facet" 
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters['allow_extrapolation'] = True
df.parameters["form_compiler"]["representation"] = 'uflacs'

# global parameters for the degree
degV = 1
degPhi = 2 + degV 

# création des frontières 

def dirichlet(point):
    return point.x() >= -10.5
def neumann(point):
    return not(dirichlet(point))

# expression of phi (replace with construction of phi when working)
class phi_expr(df.UserExpression) : 
    def eval(self, value, x):
        value[0] = -1.0/8.0 + (x[0]-0.5)**2 + (x[1]-0.5)**2 
    def value_shape(self):
        return (2,)

# build domain 
# here we compute the creation of the background mesh 
def build_domain(nbr_cells = 5, L=1.0, l=1.0):
    
    background_mesh = df.RectangleMesh(df.Point(0.0,0.0), df.Point(L,l), nbr_cells, nbr_cells)
    phi = build_phi(background_mesh)
    Cell_omega = df.MeshFunction("size_t", background_mesh, background_mesh.topology().dim())
    Cell_omega.set_all(0)
    for cell in df.cells(background_mesh):  
        v1,v2,v3 = df.vertices(cell)
        if(phi(v1.point()) <= 0.0 or phi(v2.point()) <= 0.0 or phi(v3.point()) <= 0.0 or df.near(phi(v1.point()),0.0) or df.near(phi(v2.point()),0.0) or df.near(phi(v3.point()),0.0)):
            Cell_omega[cell] = 1
    mesh = df.SubMesh(background_mesh, Cell_omega, 1) 
    return mesh

# here we create Vphi and phi using a mesh and the expression of phi
def build_phi(mesh):
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    return phi 

# we create boundaries and the measures. In the case of neumann boundaries, we create the restriction.
def build_boundaries(mesh, phi, force_mixt = False):
    
    mesh.init(1,2) 
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1) 
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())  
    cell_sub = df.MeshFunction("bool", mesh, mesh.topology().dim())
    facet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim()-1)
    vertices_sub = df.MeshFunction("bool", mesh, mesh.topology().dim()-2)
    Facet.set_all(0)
    Cell.set_all(0)
    cell_sub.set_all(0)
    facet_sub.set_all(0)
    vertices_sub.set_all(0)

    for cell in df.cells(mesh) :
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if(phi(v1.point())*phi(v2.point()) <= 0.0 or df.near(phi(v1.point())*phi(v2.point()),0.0)) : 
                # Cells for dirichlet condition
                if dirichlet(v1.point()) and dirichlet(v2.point()): 
                    Cell[cell] = 2
                    cell_sub[cell] = 1
                    for facett in df.facets(cell):  
                        Facet[facett] = 2
                        facet_sub[facett] = 1
                        v1, v2 = df.vertices(facett)
                        vertices_sub[v1], vertices_sub[v2] = 1,1
                # Cells for Neumann condition
                else : 
                    Cell[cell] = 1
                    cell_sub[cell] = 1
                    for facett in df.facets(cell):  
                        Facet[facett] = 1  
                        facet_sub[facett] = 1
                        v1, v2 = df.vertices(facett)
                        vertices_sub[v1], vertices_sub[v2] = 1,1
    nbr_dirichlet, nbr_neumann = 0,0
    for cell in df.cells(mesh):
        if Cell[cell] == 1 :
            nbr_neumann += 1
        if Cell[cell] == 2 :
            nbr_dirichlet += 1
    print("Number of cells for Neumann condition : ", nbr_neumann)
    print("Number of cells for Dirichlet condition : ", nbr_dirichlet)   
    
    dx = df.Measure("dx", mesh, subdomain_data = Cell)
    dS = df.Measure("dS", mesh, subdomain_data = Facet)
    if (nbr_dirichlet != 0 and nbr_neumann != 0) or force_mixt == False: # if there are some neumann cells, we create the measure considering Facet for ds
        ds = df.Measure("ds", mesh, subdomain_data = Facet)
    else : 
        ds = df.Measure("ds", mesh)

    if nbr_neumann != 0 or force_mixt == False:     
        File2 = df.File("sub.rtc.xml/mesh_function_2.xml")
        File2 << cell_sub
        File1 = df.File("sub.rtc.xml/mesh_function_1.xml")
        File1 << facet_sub
        File0 = df.File("sub.rtc.xml/mesh_function_0.xml")
        File0 << vertices_sub   
        yp_res = mph.MeshRestriction(mesh,"sub.rtc.xml")
        return nbr_dirichlet, nbr_neumann, dx, ds, dS, yp_res 
    
    else : 
        return nbr_dirichlet, nbr_neumann, dx, ds, dS

# for - laplace(u) + u = f
def compute_solution(u_ex):
    return -df.div(df.grad(u_ex)) + u_ex 

# compute the dirichlet boundary condition
def compute_uD(u_ex, phi):
    return u_ex * (1 + phi) 

# we compute the neumann boundary condition
def compute_g(u_ex, phi):
    return df.inner(df.grad(u_ex),df.grad(phi))/(df.inner(df.grad(phi),df.grad(phi))**0.5) + u_ex*phi

# compute l2 error
def compute_l2_error(u, u_ex, dx):
    return (df.assemble((((u_ex-u))**2)*dx)**(0.5))/(df.assemble((((u_ex))**2)*dx)**(0.5))

# compute h1 error
def compute_h1_error(u, u_ex, dx):
    return (df.assemble(((df.grad(u_ex-u))**2)*dx)**(0.5))/(df.assemble(((df.grad(u_ex))**2)*dx)**(0.5))

# phi fem for Dirichlet conditions (more efficient if Gamma = Gamma_D)
def phi_fem_dirichlet(mesh, phi, u_D, f, dx, ds, dS):
    
    V = df.FunctionSpace(mesh, 'CG', degV)
    w = df.TrialFunction(V)
    v = df.TestFunction(V)
    n = df.FacetNormal(mesh)
    h = df.CellDiameter(mesh)
    h_avg = (h('+') + h('-'))/2.0
    sigma = 20.0
    
    Gh =  sigma*h_avg*df.jump(df.grad(phi*w),n)*df.jump(df.grad(phi*v),n)*dS(2) \
        + sigma*h**2*(-df.div(df.grad(phi*w)) + phi*w)*(-df.div(df.grad(phi*v)) + phi*v)*dx(2) \
    
    Ghrhs = sigma*h**2*f*(-df.div(df.grad(phi*v)) + phi*v)*dx(2) \
        - sigma*h**2*(-df.div(df.grad(u_D)) + u_D)*(-df.div(df.grad(phi*v)) -+ phi*v)*dx(2) \
        - sigma*h_avg*df.jump(df.grad(u_D),n)*df.jump(df.grad(phi*v),n)*dS(2)  
        
    a = df.inner(df.grad(phi*w),df.grad(phi*v))*dx + phi*w*phi*v*dx - df.inner(df.grad(phi*w),n)*phi*v*ds \
        + Gh 
    L = f*phi*v*dx + Ghrhs - u_D*phi*v*dx - df.inner(df.grad(u_D),df.grad(phi*v))*dx + df.inner(df.grad(u_D),n)*phi*v*ds 

    
    A = df.assemble(a)
    B = df.assemble(L)
    w = df.Function(V)
    df.solve(A, w.vector(),B)
    u = phi*w + u_D 
    return u 

# phi fem for Neumann conditions 
def phi_fem_neumann(mesh, phi, f, g, dx, ds, dS, yp_res):
        
    V = df.FunctionSpace(mesh, "CG", degV)
    Z = df.VectorFunctionSpace(mesh,"CG",degV, dim =2)
    Q = df.FunctionSpace(mesh,"DG",degV-1)
    W = mph.BlockFunctionSpace([V,Z,Q], restrict=[None,yp_res,yp_res])
    uyp = mph.BlockTrialFunction(W)
    (u, y, p) = mph.block_split(uyp)
    vzq = mph.BlockTestFunction(W)
    (v, z, q) = mph.block_split(vzq)

    gamma_div, gamma_u, gamma_p, sigma = 1.0, 1.0, 1.0, 0.01
    h = df.CellDiameter(mesh)
    n = df.FacetNormal(mesh)
   
    # Construction of the bilinear and linear forms
    boundary_penalty = sigma*df.avg(h)*df.inner(df.jump(df.grad(u),n), df.jump(df.grad(v),n))*dS(1) \
    
    phi_abs = df.inner(df.grad(phi),df.grad(phi))**0.5

    auv = df.inner(df.grad(u), df.grad(v))*dx  + u*v*dx\
        + gamma_u*df.inner(df.grad(u),df.grad(v))*dx(1) \
        + boundary_penalty \
        + gamma_div*u*v*dx(1)

    auz = gamma_u*df.inner(df.grad(u),z)*dx(1) + gamma_div*u*df.div(z)*dx(1) 
    auq = 0.0 
    
    ayv = df.inner(df.dot(y,n),v)*ds + gamma_u*df.inner(y,df.grad(v))*dx(1)  + gamma_div*df.div(y)*v*dx(1)
        
    ayz = gamma_u*df.inner(y,z)*dx(1) + gamma_div*df.inner(df.div(y), df.div(z))*dx(1) \
        + gamma_p*h**(-2)*df.inner(df.dot(y,df.grad(phi)), df.dot(z,df.grad(phi)))*dx(1)
    ayq = gamma_p*h**(-3)*df.inner(df.dot(y,df.grad(phi)), q*phi)*dx(1)
    
    apv = 0.0
    apz = gamma_p*h**(-3)*df.inner(p*phi, df.dot(z,df.grad(phi)))*dx(1)
    apq = gamma_p*h**(-4)*df.inner(p*phi,q*phi)*dx(1) 
    
    lv = df.inner(f,v)*dx  \
        + gamma_div*f*v*dx(1)
    lz = df.inner(f, df.div(z))*dx(1) - gamma_p*h**(-2)*df.inner(g*phi_abs, df.dot(z,df.grad(phi)))*dx(1)
    lq = - gamma_p*h**(-3)*df.inner(g*phi_abs,q*phi)*dx(1) 
    
    a = [[auv,auz,auq],
         [ayv,ayz,ayq],
         [apv,apz,apq]]
    l = [lv,lz,lq]
    A = mph.block_assemble(a)
    B = mph.block_assemble(l)
    UU = mph.BlockFunction(W)
    mph.block_solve(A, UU.block_vector(), B)
    u = UU[0]
    return u

# phi fem for mixed conditions
def phi_fem_dirichlet_neumann(mesh, phi, u_D,f, g, dx, ds, dS, yp_res):
    
    V = df.FunctionSpace(mesh, "CG", degV)
    Z = df.VectorFunctionSpace(mesh,"CG",degV, dim = 2)
    Q = df.FunctionSpace(mesh,"DG",degV-1)
    W = mph.BlockFunctionSpace([V,Z,Q], restrict=[None,yp_res,yp_res])
    uyp = mph.BlockTrialFunction(W)
    (u, y, p) = mph.block_split(uyp)
    vzq = mph.BlockTestFunction(W)
    (v, z, q) = mph.block_split(vzq)

    h = df.CellDiameter(mesh)
    n = df.FacetNormal(mesh)
    gamma_div, gamma_u, gamma_p, sigma_N, sigma_D, gamma_D = 1.0, 1.0, 1.0, 0.01, 20.0, 20.0

    # Construction of the bilinear and linear forms
    boundary_penalty = sigma_N*df.avg(h)*df.inner(df.jump(df.grad(u),n), df.jump(df.grad(v),n))*dS(1) \
                     + sigma_D*df.avg(h)*df.inner(df.jump(df.grad(u),n), df.jump(df.grad(v),n))*dS(2) \
                     + sigma_D*h**2*(df.inner(- df.div(df.grad(u)) + u ,- df.div(df.grad(v)) +v ))*dx(2)
    
    phi_abs = df.inner(df.grad(phi),df.grad(phi))**0.5

    auv = df.inner(df.grad(u), df.grad(v))*dx  + u*v*dx\
        + gamma_u*df.inner(df.grad(u),df.grad(v))*dx(1) \
        + boundary_penalty \
        + gamma_D*h**(-2)*df.inner(u,v)*dx(2) \
        - df.inner(df.dot(df.grad(u),n),v)*ds(2) \
        + gamma_div*u*v*dx(1)

    auz = gamma_u*df.inner(df.grad(u),z)*dx(1) + gamma_div*u*df.div(z)*dx(1) 
    auq = - gamma_D*h**(-3)*df.dot(u,q*phi)*dx(2)  
    
    ayv = df.inner(df.dot(y,n),v)*ds(1) + gamma_u*df.inner(y,df.grad(v))*dx(1)  + gamma_div*df.div(y)*v*dx(1)
        
    ayz = gamma_u*df.inner(y,z)*dx(1) + gamma_div*df.inner(df.div(y), df.div(z))*dx(1) \
        + gamma_p*h**(-2)*df.inner(df.dot(y,df.grad(phi)), df.dot(z,df.grad(phi)))*dx(1)
    ayq = gamma_p*h**(-3)*df.inner(df.dot(y,df.grad(phi)), q*phi)*dx(1)
    
    apv = - gamma_D*h**(-3)*df.dot(v,p*phi)*dx(2) 
    apz = gamma_p*h**(-3)*df.inner(p*phi, df.dot(z,df.grad(phi)))*dx(1)
    apq = gamma_p*h**(-4)*df.inner(p*phi,q*phi)*dx(1) \
        + gamma_D*h**(-4)*df.inner(p*phi,q*phi)*dx(2)
    
    lv = df.inner(f,v)*dx  \
        + sigma_D*h**2*df.inner(f, - df.div(df.grad(v)) + v)*dx(2) \
        + gamma_D*h**(-2)*df.dot(u_D,v)*dx(2) \
        + gamma_div*f*v*dx(1)
    lz = df.inner(f, df.div(z))*dx(1) - gamma_p*h**(-2)*df.inner(g*phi_abs, df.dot(z,df.grad(phi)))*dx(1)
    lq = - gamma_p*h**(-3)*df.inner(g*phi_abs,q*phi)*dx(1) - gamma_D*h**(-3)*df.inner(u_D,q*phi)*dx(2)
    
    a = [[auv,auz,auq],
         [ayv,ayz,ayq],
         [apv,apz,apq]]
    l = [lv,lz,lq]
    A = mph.block_assemble(a)
    B = mph.block_assemble(l)
    UU = mph.BlockFunction(W)
    mph.block_solve(A, UU.block_vector(), B)
    u_h = UU[0] 
    return u_h
# we solve the phi fem on a babckground mesh defined by parameters. 
def solve_phi_fem(nbr_cells = 5, L=1.0, l=1.0, force_mixt = False):
    mesh = build_domain(nbr_cells, L, l)
    phi = build_phi(mesh)
    u_ex = df.Expression('exp(x[1])*sin(x[0])', degree = 6, domain = mesh)
    f = compute_solution(u_ex)
    boundaries = build_boundaries(mesh, phi, force_mixt)
    nbr_dirichlet = boundaries[0]
    nbr_neumann = boundaries[1]
    dx, ds, dS = boundaries[2], boundaries[3], boundaries[4]
    if nbr_neumann == 0 and force_mixt == True:
        u_D = compute_uD(u_ex, phi) 
        u = phi_fem_dirichlet(mesh, phi, u_D, f, dx, ds, dS)
    elif nbr_dirichlet == 0 :
        g = compute_g(u_ex,phi)
        u = phi_fem_neumann(mesh, phi, f, g, dx, ds, dS, boundaries[5])
    else : 
        u_D = compute_uD(u_ex, phi) 
        g = compute_g(u_ex,phi)
        u = phi_fem_dirichlet_neumann(mesh, phi,u_D, f, g, dx, ds, dS, boundaries[5])
    
    return u, u_ex, dx, mesh.hmax() # we return the solution and the measure dx in case we want to compute the error 

# we compute the standard fem (in case we want to compare the error)
def compute_standard_fem(mesh, phi, u_D, f, g):
    V = df.FunctionSpace(mesh, 'CG', degV)  
    v = df.TestFunction(V)
    u = df.TrialFunction(V)
    # Definition of the boundary to apply Dirichlet condition
    boundary = 'on_boundary && x[0] >= -0.5'
    bc = df.DirichletBC(V, u_D, boundary)
    # Resolution of the variationnal problem
    a = df.inner(df.grad(u),df.grad(v))*df.dx + u*v*df.dx
    L = f*v*df.dx + g*v*df.ds
    A = df.assemble(a)
    B = df.assemble(L)
    bc.apply(A,B) # apply Dirichlet boundary conditions to the problem
    u = df.Function(V)
    df.solve(A,u.vector(),B)
    # Compute and store h and L2 H1 errors
    return u, mesh.hmax()

# to plot the error for the phi fem
def plot_error_phi_fem(size, error_l2, error_h1, savefig = False):
    plt.figure()
    plt.loglog(size,error_h1,'o--', label=r'$\phi$-fem $H^1$ error')
    plt.loglog(size,error_l2,'o-', label=r'$\phi$-fem $L^2$ error')
    plt.loglog(size, size, '.', label="Linear")
    plt.loglog(size,[hhh**2 for hhh in size], '.',label="Quadratic")
    plt.xlabel("$h$")
    plt.ylabel(r"$\frac{\|u-u_h\|{\|u\|}$")
    plt.legend(loc='upper right')
    plt.title(r'Relative error : $\frac{\| u-u_h \|}{\|u\|} $ for $L^2$ and $H^1$ norms',y=1.025)
    if savefig :
        plt.savefig('Results/relative_error_P_{name0}.png'.format(name0=degV))
    plt.show()

# to plot errors to compare phi fem and standard fem
def plot_error_comparison_(size_phi, l2_phi, h1_phi, size_mixed, l2_mixed, h1_mixed, l2, h1, size_std, savefig = False):
    
    plt.figure()
    plt.loglog(size_phi,h1_phi,'o--', label=r'Direct $\phi$-fem  $H^1$ error')
    plt.loglog(size_phi,l2_phi,'o-', label=r'Direct $\phi$-fem  $L^2$ error')
    plt.loglog(size_mixed,h1_mixed,'x--', label=r'Mixed $\phi$-fem fem $H^1$ error')
    plt.loglog(size_mixed,l2_mixed,'x-', label=r'Mixed $\phi$-fem $L^2$ error')    
    plt.loglog(size_std,h1,'<--', label=r'Standard fem $H^1$ error')
    plt.loglog(size_std,l2,'<-', label=r'Standard fem $L^2$ error')    
    #plt.loglog(size_std,h1_std,'x-', label=r'Standard fem $H^1$ error')
    #plt.loglog(size_std,l2_std,'x-', label=r'Standard fem $L^2$ error')    
    if degV == 1 :
        plt.loglog(size_phi, size_phi, '.', label="Linear")
        plt.loglog(size_phi,[hhh**2 for hhh in size_phi], '.',label="Quadratic")
    elif degV == 2 :
        plt.loglog(size_phi,[hhh**2 for hhh in size_phi], '.',label="Quadratic")
        plt.loglog(size_phi,[hhh**3 for hhh in size_phi], '.',label="Cubic")
    plt.xlabel("$h$")
    plt.ylabel(r'$\frac{\|u-u_h\|}{\|u\|}$')
    plt.legend(loc='upper right')
    plt.title(r'Relative error : $\frac{\| u-u_h \|}{\|u\|} $ for $L^2$ and $H^1$ norms', y=1.025)
    if savefig :
        plt.savefig('Phi_fem/relative_error_P_{name0}_pure_dirichlet.png'.format(name0=degV))
    plt.show()    
def plot_error_comparison(size_phi, l2_phi, h1_phi, size_std, l2_std, h1_std, savefig = False):
    
    plt.figure()
    plt.loglog(size_phi,h1_phi,'o--', label=r'Mixed $\phi$-fem  $H^1$ error')
    plt.loglog(size_phi,l2_phi,'o-', label=r'Mixed $\phi$-fem  $L^2$ error')   
    plt.loglog(size_std,h1_std,'<--', label=r'Standard fem $H^1$ error')
    plt.loglog(size_std,l2_std,'<-', label=r'Standard fem $L^2$ error')    
    #plt.loglog(size_std,h1_std,'x-', label=r'Standard fem $H^1$ error')
    #plt.loglog(size_std,l2_std,'x-', label=r'Standard fem $L^2$ error')    
    if degV == 1 :
        plt.loglog(size_phi, size_phi, '.', label="Linear")
        plt.loglog(size_phi,[hhh**2 for hhh in size_phi], '.',label="Quadratic")
    elif degV == 2 :
        plt.loglog(size_phi,[hhh**2 for hhh in size_phi], '.',label="Quadratic")
        plt.loglog(size_phi,[hhh**3 for hhh in size_phi], '.',label="Cubic")
    plt.xlabel("$h$")
    plt.ylabel(r'$\frac{\|u-u_h\|}{\|u\|}$')
    plt.legend(loc='upper right')
    plt.title(r'Relative error : $\frac{\| u-u_h \|}{\|u\|} $ for $L^2$ and $H^1$ norms', y=1.025)
    if savefig :
        plt.savefig('Phi_fem/relative_error_P_{name0}.png'.format(name0=degV))
    plt.show()       
if __name__ == "__main__":
    convergence = False 
    comparison = True
    if convergence : 
        size, error_h1, error_l2 = [], [], []
        start, end, step = 2, 7, 1
        for i in range(start, end, step):
            print('iteration :', i)
            u, u_ex, dx, h = solve_phi_fem(nbr_cells = 10*2**i)
            size.append(h)
            #error_h1.append(compute_h1_error(u, u_ex, dx))
            error_l2.append(compute_l2_error(u, u_ex, dx))
            #print('H1 error : {:.6f}'.format(error_h1[-1]))
            print('L2 error : {:.6f}'.format(error_l2[-1]))
        plot_error_phi_fem(size, error_l2, error_h1)
    if comparison :    
        size_mixed, error_h1_mixed, error_l2_mixed = [], [], []
        size, error_h1, error_l2 = [], [], []
        size_std, error_h1_std, error_l2_std = [], [], []
        start, end, step = 0, 4, 1
        for i in range(start, end, step):
            print('iteration :', i)
            u, u_ex, dx, h = solve_phi_fem(nbr_cells = 25*2**i, force_mixt = True)
            size_mixed.append(h)
            error_h1_mixed.append(compute_h1_error(u, u_ex, dx))
            error_l2_mixed.append(compute_l2_error(u, u_ex, dx))
            print('H1 error : {:.6f}'.format(error_h1_mixed[-1]))
            print('L2 error : {:.6f}'.format(error_l2_mixed[-1]))
        
        for i in range(start, end, step):
            print('iteration :', i)
            u, u_ex, dx, h = solve_phi_fem(nbr_cells = 25*2**i, force_mixt = False)
            size.append(h)
            error_h1.append(compute_h1_error(u, u_ex, dx))
            error_l2.append(compute_l2_error(u, u_ex, dx))
            print('H1 error : {:.6f}'.format(error_h1[-1]))
            print('L2 error : {:.6f}'.format(error_l2[-1]))

    
        for i in range(start, end, step):
            print('iteration :', i)
            mesh = mshr.generate_mesh(mshr.Circle(df.Point(0.5,0.5),df.sqrt(2.0)/4.0), 25*2**(i-1))
            phi = build_phi(mesh)
            u_ex = df.Expression('exp(x[1])*sin(x[0])', degree = 6, domain = mesh)
            f = compute_solution(u_ex)
            u_D = compute_uD(u_ex, phi) 
            g = compute_g(u_ex, phi)
            u, h = compute_standard_fem(mesh, phi, u_D, f,g)
            size_std.append(h)
            error_h1_std.append(compute_h1_error(u, u_ex, df.dx))
            error_l2_std.append(compute_l2_error(u, u_ex, df.dx))
            print('H1 error : {:.6f}'.format(error_h1_std[-1]))
            print('L2 error : {:.6f}'.format(error_l2_std[-1]))
        
        plot_error_comparison_(size, error_l2, error_h1, size_mixed, error_l2_mixed, error_h1_mixed, error_l2_std, error_h1_std, size_std, savefig=True)
    

        #plot_error_comparison(size, error_l2, error_h1, size2, l2_std, h1_std, True)

        
