clear all;
close all;
format short e

%     
% Main file of Legendre Gelerkin method
% Classical method
% Two more files are needed here: lepoly.m and legslbndm.m
% -\ep u_xx - u*u_x = f

%1/2*(u^2)_x = 1/2*2*u*u_x = u*u_x

N = 32;
ep= 0.5;
tol = 1e-12;
err = 1;
it = 1;
% Compute LGL points
x = legslbndm(N+1);
D = legslbdiff(N+1,x);

% Given Boundary conditions, we can find the coefficient of Legendre
% polynomials
a(1:N+1) = 0;
b(1:N+1) = -1;


% Random force
m1 = 2*rand(1);
m2 = 2*rand(1);
w1 = 2*rand(1)-1;
w2 = 2*rand(1)-1;
force = m1*sin(2*pi*x*w1)' + m2*cos(2*pi*x*w2)';



u_old = force*0;


% Assign the linear operator
for ii=1:N-1
    k=ii-1;
    s_diag(ii) = -(4*k+6)*b(ii);
end

S = diag(s_diag);
Mass = sparse(ep*S);

while err > tol
    f = force - u_old.*((D*u_old')');
    
    
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
    
 
    % Main solver
    u = Mass\bar_f';
 
    u_sol = zeros(N+1,1);

    for ij=1:N-1
        i_ind = ij-1;
        u_sol = u_sol + u(ij)*(  lepoly(i_ind,x) + a(ij)*lepoly(i_ind+1,x) + b(ij)*lepoly(i_ind+2,x)  );
    end
    
    err = max(u_sol - u_old');
    u_old = u_sol';
    
    it = it+1;
end


it
cumulative_error = 0;


% Important: "Use only 1 test function here!!!!!!"
for l=0:4
    temp = 0;
    diffusion = -ep*(4*l+6)*(-1)*u(l+1);

    
    temp = 1/2*(u_sol.^2).*(lepolyx(l,x) - lepolyx(l+2,x));

    convection = sum(temp*2/(N*(N+1))./(lepoly(N,x).^2));
    rhs = sum(force'.*(lepoly(l,x) - lepoly(l+2,x))*2/(N*(N+1))./(lepoly(N,x).^2));
    cumulative_error = cumulative_error + abs(diffusion - convection - rhs);
end

cumulative_error

plot(x,u_sol,'-ok','linewidth',1.5)
title('Numerical solution')
xlabel('X-axis')

%DE = -ep*D^2*u_sol + D*u_sol.*u_sol - force'
