clear all;
close all;
format short e

%     
% Main file of Legendre Gelerkin method
% Classical method
% Two more files are needed here: lepoly.m and legslbndm.m


N = 64;
ep=10^-2;

% Compute LGL points
x = legslbndm(N+1);
D = legslbdiff(N+1,x);

% Given Boundary conditions, we can find the coefficient of Legendre
% polynomials
a(1:N+1) = 0;
b(1:N+1) = -1;

% External force
f(1:N+1) = 1;

% Assign the linear operator
for ii=1:N-1
    k=ii-1;
    s_diag(ii) = -(4*k+6)*b(ii);
    
    phi_k_M = D*(lepoly(k,x) + a(ii)*lepoly(k+1,x) + b(ii)*lepoly(k+2,x));
    
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

Mass = sparse(ep*S-M);

% Main solver
u = Mass\bar_f';

% Change the basis
g(1) = u(1);g(2) = u(2) + a(1)*u(1);

for i=3:N-1
    k=i-1;
    g(i) = u(i) + a(i-1)*u(i-1) + b(i-2)*u(i-2);
end
g(N) = a(N-1)*u(N-1)+b(N-2)*u(N-2);
g(N+1) = b(N-1)*u(N-1);



% Forward Legendre transform
for i=1:N+1
    sum = 0;
    for j=1:N+1
        k=j-1;
        L = lepoly(k,x);
        sum = sum + g(j)*L(i);
    end
    u(i) = sum;
end



% Plot 
plot(x,u,'-ok','linewidth',1.5)
title('Numerical solution')
xlabel('X-axis')
hold on
exact = 2*(exp(-(x+1)/ep) - 1)/(exp(-2/ep)-1) - (x+1);


plot(x,exact,'-xr')



norm(u-exact,2)/norm(exact,2)
