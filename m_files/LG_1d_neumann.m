clear all;
close all;
format short e

%     
% Main file of Legendre Gelerkin method
% Classical method
% Two more files are needed here: lepoly.m and legslbndm.m
% u_xx + k_u u = f
% Homogenous Neumann boundary conditions are used.

N = 32;
ku= 3.5;

% Compute LGL points
x = legslbndm(N+1);
D = legslbdiff(N+1,x);

% Given Boundary conditions, we can find the coefficient of Legendre
% polynomials
a(1:N+1) = 0;

for i=1:N+1
    k = i-1;
    b(i) = -k*(k+1)/((k+2)*(k+3));
end



% Random force
m1 = rand(1);
m2 = rand(1);
w1 = 2*rand(1)*pi;
w2 = 2*rand(1)*pi;
f = m1*sin(w1*x)' + m2*cos(w2*x)';

% Assign the linear operator
for ii=1:N-1
    k=ii-1;
    s_diag(ii) = -(4*k+6)*b(ii);
    
    phi_k_M = (lepoly(k,x) + a(ii)*lepoly(k+1,x) + b(ii)*lepoly(k+2,x));
    
    for jj=1:N-1
        if abs(ii-jj) <= 2
            l = jj-1;
            psi_l_M = (lepoly(l,x) + a(jj)*lepoly(l+1,x) + b(jj)*lepoly(l+2,x));
            M(jj,ii) = sum(psi_l_M.*phi_k_M*2/(N*(N+1))./(lepoly(N,x).^2));
        end
    end

end

S = diag(s_diag);

% Backward Legendre Transform
for i=1:N
    k=i-1;
    g(i) = (2*k+1)/(N*(N+1))*sum(f'.*(lepoly(k,x))./(lepoly(N,x).^2));
end
g(N+1) = 1/(N+1)*sum(f'./lepoly(N,x));



% Evaluate rhs of the weak form
for i=1:N-1
    k=i-1;
    bar_f(i) = g(i)/(k+1/2) + a(i)*g(i+1)/(k+3/2) + b(i)*g(i+2)/(k+5/2);
end

Mass = sparse(-S + ku*M);

% Main solver
u = Mass\bar_f';

u_sol = zeros(N+1,1);
for ij=1:N-1
    i_ind = ij-1;
    u_sol = u_sol + u(ij)*(  lepoly(i_ind,x) + a(ij)*lepoly(i_ind+1,x) + b(ij)*lepoly(i_ind+2,x)  );
end

plot(x,u_sol,'-ok','linewidth',1.5)
title('Numerical solution')
xlabel('X-axis')


%ODE = D^2*u_sol + ku*u_sol - f'
% Weak form

for l=0:2
    idx = l+1;
    
    test_func = (lepoly(l,x)+ b(idx)*lepoly(l+2,x));
    test_func_x = (lepolyx(l,x)+ b(idx)*lepolyx(l+2,x));
    
    diffusion = sum((D*u_sol.*test_func_x)*2/(N*(N+1))./(lepoly(N,x).^2));
    
    reaction = ku*sum((u_sol.*test_func)*2/(N*(N+1))./(lepoly(N,x).^2));
    rhs(idx) = sum(f'.*(test_func)*2/(N*(N+1))./(lepoly(N,x).^2));
    
    lhs(idx) =  - diffusion + reaction;
end

lhs - rhs


