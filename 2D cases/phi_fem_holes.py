import dolfin as df 
import matplotlib.pyplot as plt 
import multiphenics as mph 
import mshr
import time
from matplotlib import rc, rcParams 
from vedo.dolfin import plot
from vedo.plotter import closePlotter
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

# degree of interpolation for V and Vphi
degV = 1
degPhi = 2 + degV

# Function used to write in the outputs files
def output_latex(f,A,B):
	for i in range(len(A)):
		f.write('(')
		f.write(str(A[i]))
		f.write(',')
		f.write(str(B[i]))
		f.write(')\n')
	f.write('\n')
"""
We define the level-sets function phi :
Here we consider the case of circle centered in (0.5,0.5) of radius sqrt(2)/4, so the level-sets is the equation of the circle.
"""
class phi1_expr(df.UserExpression) : 
    def eval(self, value, x):
        value[0] = -0.15**2 + (x[0]-0.45)**2 + (x[1]-0.45)**2 

    def value_shape(self):
        return (2,)
class phi2_expr(df.UserExpression) : 
    def eval(self, value, x):
        value[0] = -0.15**2 + (x[0]-0.75)**2 + (x[1]-0.75)**2 

    def value_shape(self):
        return (2,)
# We create the lists that we'll use to store errors and computation time for the phi-fem and standard fem
Time_assemble_phi, Time_solve_phi, Time_total_phi, error_l2_phi, error_h1_phi, hh_phi = [], [], [], [], [], []
Time_assemble_standard, Time_solve_standard, Time_total_standard, error_h1_standard, error_l2_standard,  hh_standard = [], [], [], [], [], []

# we compute the phi-fem for different sizes of cells
start,end,step = 1,5,1
for i in range(start,end,step): 
    print("Phi-fem iteration : ", i)
    # we define parameters and the "global" domain O
    H = 8*2**(i)
    square = df.UnitSquareMesh(H,H)
    
    # We now define Omega using phi
    V_phi = df.FunctionSpace(square, "CG", degPhi)
    phi1 = phi1_expr(element = V_phi.ufl_element())
    phi1 = df.interpolate(phi1, V_phi)
    phi2 = phi2_expr(element = V_phi.ufl_element())
    phi2 = df.interpolate(phi2, V_phi)
    trou1, trou2, trou12 = 4,5, 6 
    Cell_omega = df.MeshFunction("size_t", square, 2)
    Cell_omega.set_all(0)
    for cell in df.cells(square):  
        v1,v2,v3 = df.vertices(cell)
        if(phi1(v1.point()) >= 0.0 or phi1(v2.point()) >= 0.0 or phi1(v3.point()) >= 0.0 or df.near(phi1(v1.point()),0.0) or df.near(phi1(v2.point()),0.0) or df.near(phi1(v3.point()),0.0)):
            if(phi2(v1.point()) >= 0.0 or phi2(v2.point()) >= 0.0 or phi2(v3.point()) >= 0.0 or df.near(phi2(v1.point()),0.0) or df.near(phi2(v2.point()),0.0) or df.near(phi2(v3.point()),0.0)):
                Cell_omega[cell] = 1
    mesh = df.SubMesh(square, Cell_omega, 1)   
    #plot(mesh)
    hh_phi.append(mesh.hmax()) # store the size of each element for this iteration  

    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi1 = phi1_expr(element = V_phi.ufl_element())
    phi1 = df.interpolate(phi1, V_phi)
    phi2 = phi2_expr(element = V_phi.ufl_element())
    phi2 = df.interpolate(phi2, V_phi)
    # Selection of cells and facets on the boundary
    mesh.init(1,2) 
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1) 
    Cell = df.MeshFunction("size_t", mesh, 2)  
    cell_sub = df.MeshFunction("bool", mesh, 2)
    facet_sub = df.MeshFunction("bool", mesh, 1)
    vertices_sub = df.MeshFunction("bool", mesh, 0)
    Facet.set_all(0)
    Cell.set_all(0)
    cell_sub.set_all(0)
    facet_sub.set_all(0)
    vertices_sub.set_all(0)

    for cell in df.cells(mesh) :
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if(phi1(v1.point())*phi1(v2.point()) <= 0.0 or df.near(phi1(v1.point())*phi1(v2.point()),0.0)) : 
                Cell[cell] = trou1  
                cell_sub[cell] = 1
                for facett in df.facets(cell):  
                    Facet[facett] = trou1 
                    facet_sub[facett] = 1
                    v1, v2 = df.vertices(facett)
                    vertices_sub[v1], vertices_sub[v2] = 1,1
            if(phi2(v1.point())*phi2(v2.point()) <= 0.0 or df.near(phi2(v1.point())*phi2(v2.point()),0.0)) : 
                if Cell[cell] == trou1 :
                    Cell[cell] = trou12
                    cell_sub[cell] = 1
                    for facett in df.facets(cell):  
                        Facet[facett] = trou2 
                        facet_sub[facett] = 1
                        v1, v2 = df.vertices(facett)
                        vertices_sub[v1], vertices_sub[v2] = 1,1
                else :
                    Cell[cell] = trou2  
                    cell_sub[cell] = 1
                    for facett in df.facets(cell):  
                        Facet[facett] = trou2 
                        facet_sub[facett] = 1
                        v1, v2 = df.vertices(facett)
                        vertices_sub[v1], vertices_sub[v2] = 1,1
            
    
    File2 = df.File("sub.rtc.xml/mesh_function_2.xml")
    File2 << cell_sub
    File1 = df.File("sub.rtc.xml/mesh_function_1.xml")
    File1 << facet_sub
    File0 = df.File("sub.rtc.xml/mesh_function_0.xml")
    File0 << vertices_sub
    trou11 = df.SubMesh(mesh, Cell, trou1)
    trou22 = df.SubMesh(mesh, Cell, trou2)
    if i == end-1 :
        plot(mesh, c='white', add = True)
        plot(trou11, c= 'purple', add = True)
        plot(trou22, c = 'red', add = True)
        closePlotter()
    
    # Beginning of variationnal problem resolution
    yp_res = mph.MeshRestriction(mesh,"sub.rtc.xml")
    V = df.FunctionSpace(mesh, "CG", degV)
    Z = df.VectorFunctionSpace(mesh,"CG",degV, dim = 2)
    Q = df.FunctionSpace(mesh,"DG",degV-1)
    W = mph.BlockFunctionSpace([V,Z,Q], restrict=[None,yp_res,yp_res])
    uyp = mph.BlockTrialFunction(W)
    (u, y, p) = mph.block_split(uyp)
    vzq = mph.BlockTestFunction(W)
    (v, z, q) = mph.block_split(vzq)

    dx = df.Measure("dx", mesh, subdomain_data = Cell)
    ds = df.Measure("ds", mesh, subdomain_data= Facet)
    dS = df.Measure("dS", mesh, subdomain_data = Facet)
    areaOm1 = df.assemble(1.*dx(trou1))
    areaOm2 = df.assemble(1.*dx(trou2))
    print("area(trou1) = ", areaOm1)
    print("area(trou2) = ", areaOm2)
    gamma_div, gamma_u, gamma_p, sigma = 1.0, 1.0, 1.0, 0.01
    h = df.CellDiameter(mesh)
    n = df.FacetNormal(mesh)
    u_D = df.Constant(0.0)
    f = df.Expression('-9.81 * x[1]', degree=4, domain = mesh)
    g = df.Constant(0.0)
    
    # Construction of the bilinear and linear forms
    boundary_penalty = sigma*df.avg(h)*df.inner(df.jump(df.grad(u),n), df.jump(df.grad(v),n))*dS(trou1) \
                     + sigma*h**2*(df.inner(df.div(df.grad(u)), df.div(df.grad(v))))*dx(trou1) \
                     + sigma*df.avg(h)*df.inner(df.jump(df.grad(u),n), df.jump(df.grad(v),n))*dS(trou2) \
                     + sigma*h**2*(df.inner(df.div(df.grad(u)), df.div(df.grad(v)) ))*dx(trou2) \
                     + sigma*df.avg(h)*df.inner(df.jump(df.grad(u),n), df.jump(df.grad(v),n))*dS(trou12) \
                     + sigma*h**2*(df.inner(df.div(df.grad(u)), df.div(df.grad(v)) ))*dx(trou12) 
                     
    phi1_abs = df.inner(df.grad(phi1),df.grad(phi1))**0.5
    phi2_abs = df.inner(df.grad(phi2),df.grad(phi2))**0.5

    auv = df.inner(df.grad(u), df.grad(v))*dx  \
        + gamma_u*df.inner(df.grad(u),df.grad(v))*dx(trou1) + gamma_u*df.inner(df.grad(u),df.grad(v))*dx(trou2) \
        + gamma_u*df.inner(df.grad(u),df.grad(v))*dx(trou12) \
        + boundary_penalty 

    auz = gamma_u*df.inner(df.grad(u),z)*dx(trou1) \
        + gamma_u*df.inner(df.grad(u),z)*dx(trou2)  \
        + gamma_u*df.inner(df.grad(u),z)*dx(trou12)  

    auq = 0.0 
    
    ayv = df.inner(df.dot(y,n),v)*ds(trou1) + df.inner(df.dot(y,n),v)*ds(trou2) \
        + df.inner(df.dot(y,n),v)*ds(trou12) \
        + gamma_u*df.inner(y,df.grad(v))*dx(trou1) \
        + gamma_u*df.inner(y,df.grad(v))*dx(trou2) \
        + gamma_u*df.inner(y,df.grad(v))*dx(trou12)  


    ayz = gamma_u*df.inner(y,z)*dx(trou1) + gamma_div*df.inner(df.div(y), df.div(z))*dx(trou1) \
        + gamma_p*h**(-2)*df.inner(df.dot(y,df.grad(phi1)), df.dot(z,df.grad(phi1)))*dx(trou1) \
        + gamma_u*df.inner(y,z)*dx(trou2) + gamma_div*df.inner(df.div(y), df.div(z))*dx(trou2) \
        + gamma_p*h**(-2)*df.inner(df.dot(y,df.grad(phi2)), df.dot(z,df.grad(phi2)))*dx(trou2) \
        + gamma_u*df.inner(y,z)*dx(trou12) + gamma_div*df.inner(df.div(y), df.div(z))*dx(trou12) \
        + gamma_p*h**(-2)*df.inner(df.dot(y,df.grad(phi2)), df.dot(z,df.grad(phi2)))*dx(trou12)
    ayq = gamma_p*h**(-3)*df.inner(df.dot(y,df.grad(phi1)), q*phi1)*dx(trou1) \
        + gamma_p*h**(-3)*df.inner(df.dot(y,df.grad(phi2)), q*phi2)*dx(trou2) \
        + gamma_p*h**(-3)*df.inner(df.dot(y,df.grad(phi2)), q*phi2)*dx(trou12) 
    
    apv = 0.0
    apz = gamma_p*h**(-3)*df.inner(p*phi1, df.dot(z,df.grad(phi1)))*dx(trou1) \
        + gamma_p*h**(-3)*df.inner(p*phi2, df.dot(z,df.grad(phi2)))*dx(trou2) \
        + gamma_p*h**(-3)*df.inner(p*phi2, df.dot(z,df.grad(phi2)))*dx(trou12)
    apq = gamma_p*h**(-4)*df.inner(p*phi1,q*phi1)*dx(trou1) + gamma_p*h**(-4)*df.inner(p*phi2,q*phi2)*dx(trou2) \
        + gamma_p*h**(-4)*df.inner(p*phi2,q*phi2)*dx(trou12)  
    
    lv = df.inner(f,v)*dx + df.inner(g,v)*ds \
        + sigma*h**2*df.inner(f, - df.div(df.grad(v)))*dx(trou1) \
        + sigma*h**2*df.inner(f, - df.div(df.grad(v)))*dx(trou2) \
        + sigma*h**2*df.inner(f, - df.div(df.grad(v)))*dx(trou12) 
    lz = df.inner(f, df.div(z))*dx(trou1) \
        + df.inner(f, df.div(z))*dx(trou2) \
        + df.inner(f, df.div(z))*dx(trou12)

    lq = 0.0 
    
    a = [[auv,auz,auq],
         [ayv,ayz,ayq],
         [apv,apz,apq]]
    l = [lv,lz,lq]
    start_assemble = time.time()
    A = mph.block_assemble(a)
    B = mph.block_assemble(l)
    end_assemble = time.time()
    Time_assemble_phi.append(end_assemble-start_assemble)
    def boundary(x, on_boundary):
        return on_boundary and (df.near(x[1],0.0))
    bc1 = mph.DirichletBC(W.sub(0), u_D, boundary) # apply DirichletBC on Omega1
    bcs = mph.BlockDirichletBC([bc1])
    bcs.apply(A)
    bcs.apply(B)
    UU = mph.BlockFunction(W)
    start_solve = time.time()
    mph.block_solve(A, UU.block_vector(), B)
    end_solve = time.time()
    Time_solve_phi.append(end_solve-start_solve)
    Time_total_phi.append(Time_assemble_phi[-1] + Time_solve_phi[-1])
    u_h_phi = UU[0]
    if i == end - 1 :
        plot(u_h_phi, N = 2, at=0, text ='u_h phi-FEM')
    # Compute and store relative error for H1 and L2 norms
    domain = mshr.Rectangle(df.Point(0,0), df.Point(1,1)) \
        - mshr.Circle(df.Point(0.45,0.45), 0.15) - mshr.Circle(df.Point(0.75,0.75), 0.15)
    H = 8*2**(4) # to have approximately the same precision as in the phi-fem computation
    mesh = mshr.generate_mesh(domain,H)
    print("Standard fem iteration : ", i)
    # FunctionSpace P1
    V = df.FunctionSpace(mesh, 'CG', degV)  
    v = df.TestFunction(V)
    u = df.TrialFunction(V)
    u_D = df.Constant(0.0) 
    g = df.Constant(0.0)
    f = df.Expression('-9.81*x[1]', degree=4, domain = mesh)
    # Resolution of the variationnal problem
    def boundary(x, on_boundary):
        return on_boundary and (df.near(x[1],0.0))
    bc = df.DirichletBC(V, u_D, boundary)
    a = df.inner(df.grad(u), df.grad(v))*df.dx 
    l = df.inner(f,v)*df.dx + df.inner(g,v)*df.ds
    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(l)
    end_assemble = time.time()
    Time_assemble_standard.append(end_assemble-start_assemble)
    bc.apply(A, B)
    start_standard = time.time()
    u = df.Function(V)
    df.solve(A,u.vector(),B)
    end_standard = time.time()
    u_h = u
    Time_solve_standard.append(end_standard-start_standard)
    Time_total_standard.append(Time_assemble_standard[-1] + Time_solve_standard[-1])
    # Compute and store h and L2 H1 errors
    hh_standard.append(mesh.hmax())
    if i == end - 1 :
        plot(u_h, at = 1, text = 'u_h standard')
    u_h_phi = df.project(u_h_phi, V)
    error_l2_phi.append((df.assemble((((u_h-u_h_phi))**2)*df.dx)**(0.5))/(df.assemble((((u_h))**2)*df.dx)**(0.5)))            
    error_h1_phi.append((df.assemble(((df.grad(u_h- u_h_phi))**2)*df.dx)**(0.5))/(df.assemble(((df.grad(u_h))**2)*df.dx)**(0.5)))

# Computation of the standard FEM       
# Plot results : error/precision, Time/precision, Time/error and Total_time/error

plt.figure()
plt.loglog(hh_phi,error_h1_phi,'o--', label=r'$\phi$-FEM $H^1$')
plt.loglog(hh_phi,error_l2_phi,'o-', label=r'$\phi$-FEM $L^2$')
plt.loglog(hh_phi, hh_phi, '.', label="Linear")
plt.loglog(hh_phi,[hhh**2 for hhh in hh_phi], '.',label="Quadratic")
plt.xlabel("$h$")
plt.ylabel(r'$\frac{\|u-u_h\|}{\|u\|}$')
plt.legend(loc='upper right', ncol=2)
plt.title(r'Relative error : $ \frac{\|u-u_h\|}{\|u\|} $ for $L^2$ and $H^1$ norms', y=1.025)
plt.tight_layout()
#plt.savefig('relative_error_P_{name0}.png'.format(name0=degV))
plt.show()
