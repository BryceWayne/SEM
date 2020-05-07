clear all;
close all
% Ep=10^-4
%X1 = [4 8 16 32];
%Y = [4.3453e-13 3.3939e-13 7.3803e-16 3.1300e-15];


% Ep = 10^-8
X1 = [20 40 80 160];
%Y =   [3.3976e-08 4.2954e-09 1.2734e-09 8.9471e-10];


Y = [   4.2104e-03
   2.6140e-04
   1.6516e-05
   5.0090e-06];


semilogy(X1,Y,'-ks','LineWidth',2,'MarkerSize',7);
xlabel('N: Number of basis');
ylabel('Relative $L^2$ error','Interpreter','latex');
title('{Relative $L^2$ error versus number of basis with $\epsilon = 10^{-9}$}','FontSize',15,'Interpreter','latex')
%l = legend('$\epsilon$ = 0.05','$\epsilon$ = 0.025','$\epsilon$ = 0.0125');
%set(l,'Interpreter','Latex')
%legend('\epsilon = 0.05','\epsilon = 0.025','\epsilon = 0.0125','Interpreter','latex')
%axis([])
%ylim([10^(-16) 1])




