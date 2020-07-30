clear all;
close all;
format short e

% July.06.2020.    
% Main file of Legendre Gelerkin method
% Classical method
% Two more files are needed here: lepoly.m and legslbndm.m
% u_xx + k_u u = f
% Homogenous Neumann boundary conditions are used.

N = 4;
ku= 3.5;

% Compute LGL points
x = legslbndm(N+1);
D = legslbdiff(N+1,x);

% Given Boundary conditions, we can find the coefficient of Legendre
% polynomials
a(1:N+1) = 0;


%
%  As we are working on the Neumann condition, the coefficient b is not s
%  constant anymore.
%
%
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

f = ku*cos(2*pi*x) - 4*pi^2*cos(2*pi*x);
f = f';
% Assign the linear operator
for ii=1:N-1
    k=ii-1;
    s_diag(ii) = -(4*k+6)*b(ii);
    
    %
    % We do not use lepolyx here as there is no convective term
    %
    phi_k_M = (lepoly(k,x) + a(ii)*lepoly(k+1,x) + b(ii)*lepoly(k+2,x));
%     phi_k_M
    for jj=1:N-1
        if abs(ii-jj) <= 2
            l = jj-1;
            psi_l_M = (lepoly(l,x) + a(jj)*lepoly(l+1,x) + b(jj)*lepoly(l+2,x));
%             psi_l_M
            M(jj,ii) = sum(psi_l_M.*phi_k_M*2/(N*(N+1))./(lepoly(N,x).^2));
            M(jj,ii)
        end
    end

end
M
S = diag(s_diag);
S


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

%
% Note that there is negative sign infromt of S as there is no - in front
% of diffusive term.
%
Mass = sparse(-S + ku*M);

% Main solver
u = Mass\bar_f';

u_sol = zeros(N+1,1);
for ij=1:N-1
    i_ind = ij-1;
    u_sol = u_sol + u(ij)*(  lepoly(i_ind,x) + a(ij)*lepoly(i_ind+1,x) + b(ij)*lepoly(i_ind+2,x)  );
end


u_sol
return
plot(x,u_sol,'-ok','linewidth',1.5)
title('Numerical solution')
xlabel('X-axis')

return
% % Weak form
% 
% cumulative_lhs = 0;
% cumulative_rhs = 0;
% 
% for l=0:2
%     idx = l+1;
%     test_func = (lepoly(l,x)+ b(idx)*lepoly(l+2,x));
%     test_func_x = (lepolyx(l,x)+ b(idx)*lepolyx(l+2,x));
%     
%     diffusion = sum((D*u_sol.*test_func_x)*2/(N*(N+1))./(lepoly(N,x).^2));
%     
%     reaction = ku*sum((u_sol.*test_func)*2/(N*(N+1))./(lepoly(N,x).^2));
%     rhs = sum(f'.*(test_func)*2/(N*(N+1))./(lepoly(N,x).^2));
%     lhs = - diffusion + reaction;
%     
%     cumulative_rhs = cumulative_rhs + abs(rhs);
%     cumulative_lhs = cumulative_lhs + abs(lhs);
%     diffusion - diffusion2
% end
% 
% cumulative_lhs - cumulative_rhs


