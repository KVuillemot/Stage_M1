import dolfin as df 
import matplotlib.pyplot as plt 
import mshr
import time
from vedo.dolfin import plot, interactive 
from matplotlib import rc, rcParams 

plt.style.use('bmh') 
params = {'axes.labelsize': 28,
          'font.size': 22,
          'axes.titlesize': 28,
          'legend.fontsize': 20,
          'figure.titlesize': 26,
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

# degree of interpolation for V and Vphi
degV = 1
degPhi = 1 + degV 

# functions and parameters for elasticity
def sigma(u):
    return lambda_ * df.div(u)*df.Identity(2) + 2.0*mu*epsilon(u)

def epsilon(u):
    return (1.0/2.0)*(df.grad(u) + df.grad(u).T)

lambda_ = 1.25
mu = 1.0
rho = 1.0

# expression of phi
class phi_expr(df.UserExpression) : 
    def eval(self, value, x):
        value[0] = -1.0/8.0 + (x[0]-0.5)**2 + (x[1]-0.5)**2 
    def value_shape(self):
        return (2,)

# creation of the lists for the phi-fem
hh_phi, error_l2_phi, error_h1_phi = [], [], []
Time_assemble_phi, Time_solve_phi, Time_total_phi = [], [], [], 
start, end = 1, 5
#compute phi-fem
for i in range(start,end):
    H = 8*2**i
    print('phi fem itération ', i) 
    square = df.UnitSquareMesh(H,H)

    # We now define Omega using phi
    V_phi = df.FunctionSpace(square, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    Cell_omega = df.MeshFunction("size_t", square, 2)
    Cell_omega.set_all(0)
    for cell in df.cells(square):  
        v1,v2,v3 = df.vertices(cell)
        if(phi(v1.point()) <= 0.0 or phi(v2.point()) <= 0.0 or phi(v3.point()) <= 0.0 or df.near(phi(v1.point()),0.0) or df.near(phi(v2.point()),0.0) or df.near(phi(v3.point()),0.0)):
            Cell_omega[cell] = 1
    mesh = df.SubMesh(square, Cell_omega, 1) 

    # Creation of the FunctionSpace for Phi on the new mesh
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)

    # Selection of cells and facets on the boundary
    mesh.init(1,2) 
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # codimension 1
    Cell = df.MeshFunction("size_t", mesh, 2)  # codimension 0
    Facet.set_all(0)
    Cell.set_all(0)
    for cell in df.cells(mesh) : 
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if(phi(v1.point())*phi(v2.point()) <= 0.0 or df.near(phi(v1.point())*phi(v2.point()),0.0)) : # si on est sur le bord 
                Cell[cell] = 1  
                for facett in df.facets(cell):
                    Facet[facett] = 1 

    # Variationnal problem resolution
    V = df.VectorFunctionSpace(mesh, 'CG', degV)
    u_ex = df.Expression(('sin(x[0]) * exp(x[1])', 'sin(x[1]) * exp(x[0])'), degree = 6, domain = mesh)
    u_D = u_ex * (1+ phi)
    f = - df.div(sigma(u_ex))
    w = df.TrialFunction(V)
    v = df.TestFunction(V)
    n = df.FacetNormal(mesh)
    h = df.CellDiameter(mesh)
    h_avg = (h('+') + h('-'))/2.0

    dx = df.Measure("dx", mesh, subdomain_data = Cell)
    ds = df.Measure("ds", mesh)
    dS = df.Measure("dS", mesh, subdomain_data = Facet)

    sigma_ = 20.0 # Stabilization parameter
    # Creation of the bilinear and linear forms using stabilization terms and boundary condition
    a = df.inner(sigma(phi*w), epsilon(phi*v))*dx \
        - df.inner(df.dot(sigma(phi*w),n), phi*v)*ds \
        + sigma_*h_avg*df.dot(df.jump(sigma(phi*w),n),df.jump(sigma(phi*v),n))*dS(1) \
        + sigma_*h**2*df.dot(df.div(sigma(phi*w)),df.div(sigma(phi*v)))*dx(1) 

    L = df.inner(f,phi*v)*dx \
        - sigma_*h**2*df.dot(f, df.div(sigma(phi*v)))*dx(1) \
        - sigma_*h_avg*df.dot(df.jump(sigma(u_D), n), df.jump(sigma(phi*v), n))*dS(1) \
        - sigma_*(h**2)*df.dot(df.div(sigma(u_D)), df.div(sigma(phi*v)))*dx(1) \
        - df.inner(sigma(u_D), epsilon(phi*v))*dx \
        + df.inner(df.dot(sigma(u_D), n), phi*v)*ds
        
    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(L)
    end_assemble = time.time()
    Time_assemble_phi.append(end_assemble-start_assemble)
    w_h = df.Function(V)
    start_solve = time.time()
    df.solve(A, w_h.vector(),B)
    end_solve = time.time()
    Time_solve_phi.append(end_solve-start_solve)
    Time_total_phi.append(Time_assemble_phi[-1] + Time_solve_phi[-1])

    u_h = phi * w_h + u_D
    u_h = df.project(u_h,V)

    errExactL2 = df.assemble(df.inner(u_ex,u_ex)*dx(0))**0.5 
    error_L2 = df.assemble(df.inner((u_ex-u_h),(u_ex-u_h))*dx(0))**0.5 / errExactL2

    semiExactH1 = df.assemble(df.inner(df.grad(u_ex), df.grad(u_ex))*dx(0))**0.5
    semi_H1 = df.assemble(df.inner(df.grad(u_ex-u_h), df.grad(u_ex-u_h))*dx(0))**0.5 / semiExactH1
    error_l2_phi.append(error_L2)
    error_h1_phi.append(semi_H1)
    #error_l2_phi.append((df.assemble((((u_ex-u_h))**2)*df.dx)**(0.5))/(df.assemble((((u_ex))**2)*df.dx)**(0.5)))
    #error_h1_phi.append((df.assemble(((df.grad(u_ex-u_h))**2)*df.dx)**(0.5))/(df.assemble(((df.grad(u_ex))**2)*df.dx)**(0.5)))    
    hh_phi.append(mesh.hmax())
    
# standard fem with the same parameters
hh, error_l2_standard, error_h1_standard = [], [], []
Time_assemble_standard, Time_solve_standard, Time_total_standard= [], [], [], 

for i in range(start,end):
    print('standard fem iteration ', i)
    domain = mshr.Circle(df.Point(0.5,0.5), df.sqrt(2)/4)
    mesh = mshr.generate_mesh(domain, 8*2**(i-1))
    u_ex = df.Expression(('sin(x[0]) * exp(x[1])', 'sin(x[1]) * exp(x[0])'), degree = 6, domain = mesh)
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    u_D = u_ex*(1+phi)
    f = - df.div(sigma(u_ex))
    V = df.VectorFunctionSpace(mesh, 'CG', degV)
    boundary = 'on_boundary'
    u_D = u_ex * ( 1 + phi)
    bc = df.DirichletBC(V, u_D, boundary)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    n = df.FacetNormal(mesh)
    a = df.inner(sigma(u), epsilon(v))*df.dx - df.inner(df.dot(sigma(u),n), v)*df.ds
    L = df.inner(f,v)*df.dx 
    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(L)
    end_assemble = time.time()
    Time_assemble_standard.append(end_assemble-start_assemble)
    bc.apply(A,B) # apply Dirichlet boundary conditions to the problem
    start_standard = time.time()
    u = df.Function(V)
    df.solve(A,u.vector(),B)
    end_standard = time.time()
    Time_solve_standard.append(end_standard-start_standard)
    Time_total_standard.append(Time_assemble_standard[-1] + Time_solve_standard[-1])
    #plot(u, at = 1, warpZfactor = 4, interactive=False, text = r'standard fem with ' + '\n' + 'h = ' + str(mesh.hmax())) 
    
    errExactL2 = df.assemble(df.inner(u_ex,u_ex)*df.dx)**0.5 
    error_L2 = df.assemble(df.inner((u_ex-u),(u_ex-u))*df.dx)**0.5 / errExactL2

    semiExactH1 = df.assemble(df.inner(df.grad(u_ex), df.grad(u_ex))*df.dx)**0.5
    semi_H1 = df.assemble(df.inner(df.grad(u_ex-u), df.grad(u_ex-u))*df.dx)**0.5 / semiExactH1
    error_l2_standard.append(error_L2)
    error_h1_standard.append(semi_H1)
    
    #error_l2_standard.append((df.assemble((((u_ex-u))**2)*df.dx)**(0.5))/(df.assemble((((u_ex))**2)*df.dx)**(0.5)))
    #error_h1_standard.append((df.assemble(((df.grad(u_ex-u))**2)*df.dx)**(0.5))/(df.assemble(((df.grad(u_ex))**2)*df.dx)**(0.5)))    
    hh.append(mesh.hmax())
    print('L2 error = ', error_l2_standard[-1])
    #interactive()
    
#plot solutions 
"""
plot(u_h, N=3, at =0, warpZfactor = 0.4, text = 'phi-fem')
plot(u,at = 1, warpZfactor = 0.4, text = 'standard fem')
plot(mesh, u_ex,at = 2, warpZfactor = 0.4, text = 'exact solution')
"""
plt.figure()
plt.loglog(hh_phi,error_h1_phi,'o--', label=r'$\phi$-FEM $H^1$')
plt.loglog(hh_phi,error_l2_phi,'o-', label=r'$\phi$-FEM $L^2$')
plt.loglog(hh,error_h1_standard, '--x',label=r'Std FEM $H^1$')
plt.loglog(hh,error_l2_standard, '-x',label=r'Std FEM $L^2$')
if degV == 1 :
    plt.loglog(hh_phi, hh_phi, '.', label="Linear")
    plt.loglog(hh_phi,[hhh**2 for hhh in hh_phi], '.',label="Quadratic")
elif degV == 2 :
    plt.loglog(hh_phi,[hhh**2 for hhh in hh_phi], '.',label="Quadratic")
    plt.loglog(hh_phi,[hhh**3 for hhh in hh_phi], '.',label="Cubic")

plt.xlabel("$h$")
plt.ylabel(r'$\frac{\|u-u_h\|}{\|u\|}$')
plt.legend(loc='upper right', ncol=2)
plt.title(r'Relative error : $ \frac{\|u-u_h\|}{\|u\|} $ for $L^2$ and $H^1$ norms', y=1.025)
plt.tight_layout()
plt.savefig('Elasticity_Dirichlet/relative_error_P_{name0}.png'.format(name0=degV))
plt.show()

plt.figure()
plt.loglog(hh_phi,Time_assemble_phi, '-o',label=r'Assemble $\phi$-FEM')
plt.loglog(hh_phi,Time_solve_phi,'--o', label=r'Solve $\phi$-FEM')
plt.loglog(hh,Time_assemble_standard, '-x',label=r'Assemble standard FEM')
plt.loglog(hh,Time_solve_standard,'--x', label=r'Solve standard FEM')
plt.xlabel("$h$")
plt.ylabel("Time (s)")
plt.legend(loc='upper right')
plt.title("Computing time")
plt.tight_layout()
plt.savefig('Elasticity_Dirichlet/Time_precision_P_{name0}.png'.format(name0=degV))
plt.show()
plt.figure()
plt.loglog(error_l2_phi,Time_assemble_phi, '-o',label=r'Assemble $\phi$-fem')
plt.loglog(error_l2_phi,Time_solve_phi,'--o', label=r'Solve $\phi$-fem')
plt.loglog(error_l2_standard,Time_assemble_standard,'-x', label="Assemble standard FEM")
plt.loglog(error_l2_standard,Time_solve_standard,'--x', label="Solve standard FEM")
plt.xlabel(r'$\frac{\|u-u_h\|_{L^2}}{\|u\|_{L^2}}$')
plt.ylabel("Time (s)")
plt.title(r'Computing time')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Elasticity_Dirichlet/Time_error_P_{name0}.png'.format(name0=degV))
plt.show()
plt.figure()
plt.loglog(error_l2_phi,Time_total_phi,'-o', label=r'$\phi$-fem')
plt.loglog(error_l2_standard,Time_total_standard,'-x', label="Standard FEM")
plt.xlabel(r'$\frac{\|u-u_h\|_{L^2}}{\|u\|_{L^2}}$')
plt.ylabel("Time (s)")
plt.title(r'Computing time')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Elasticity_Dirichlet/Total_time_error_P_{name0}.png'.format(name0=degV))
plt.show()
