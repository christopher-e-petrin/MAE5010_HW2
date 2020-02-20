% Code by Christopher E. Petrin
% Prepared for MAE 5010: Data Assimilation (Spring 2020, Dr. Omer San)
% For Homework #2, Problem #3

clear
% INPUTS
sigma = 10;
rho = 28;
beta = 8/3;
var = 0.09;
%The assignment says dt =0.05, but that timestep is numerically unstable.
%	Therefore, selecting something smaller.
dtT = 0.01;
dtO = 2*dtT;
tf = 2;
tf_Traj = 5;
x0T = [1.0, 1.0, 1.0]';
x0O = [1.1, 1.1, 1.1]';
tic
%Part A: Compute model
syms f1(xs,ys,zs) f2(xs,ys,zs) f3(xs,ys,zs)
f1(xs,ys,zs) = -sigma*(xs-ys);
f2(xs,ys,zs) = rho*xs - ys - xs*zs;
f3(xs,ys,zs) = xs*ys - beta*zs;
fM(1,1) = f1;	fM(2,1) = f2;	fM(3,1) = f3;
M1(xs,ys,zs) = xs + dtT*f1(xs,ys,zs);
M2(xs,ys,zs) = ys + dtT*f2(xs,ys,zs);
M3(xs,ys,zs) = zs + dtT*f3(xs,ys,zs);
MM(1,1) = M1;	MM(2,1) = M2;	MM(3,1) = M3;
xSYM(1,1) = xs;	xSYM(2,1) = ys;	xSYM(3,1) = zs;
toc
%Part B: Compute Jacobian matrix
for i = 1:3
	for j = 1:3
		dM(i,j) = diff(MM(i),xSYM(j));
	end
end
toc
%Part C: Compute a true trajectory using dt = 0.05 and t max = 2.
NT = tf_Traj/dtT + 1;
xT(:,1) = x0T;
for k = 1:NT
	for i = 1:3
		% Ugly but faster way of finding model operation
		if i == 1
			M = M1;
		elseif i == 2
			M = M2;
		elseif i == 3
			M = M3;
		end
% 		M = symfun(MM(i),[xs ys zs]);
		xT(i,k+1) = M(xT(1,k),xT(2,k),xT(3,k));
	end
end
toc
clear M
%Part D: Generate observations for 4 different variances:
%	var = 0.0001, 0.01, 0.04, 0.09
N = tf/(dtO);
xO(:,1) = x0O;
for k = 1:N
	for i = 1:3
		% Ugly but faster way of finding model operation
		if i == 1
			f = f1;
		elseif i == 2
			f = f2;
		elseif i == 3
			f = f3;
		end
% 		f = symfun(fM(i),[xs ys zs]);
		xO(i,k+1) = xO(i,k) + dtO*f(xO(1,k),xO(2,k),xO(3,k));
		v(i,k) = 0 + var.*randn(1);
		z(i,k) = xO(i,k) + v(i,k);
	end
end
clear M
toc
% Outer conjugate gradient loop
for j = 1:N 
	% Iterate through each dimension
	for dim = 1:3
		
	end
end
