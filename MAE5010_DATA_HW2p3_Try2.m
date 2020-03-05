%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% MAE 5010: HW#2, P3 (Spring 2020, for Dr. Omer San)
% - Christopher E. Petrin
% - Objective: Use 4D-VAR to solve Lorenz equations and optimize given
%   erroneous data. 
% --- Discretization: 1st-order Euler forward
% --- Optimizaiton: Conjugate gradient (CG) method
% - Last edited: 4-Mar-2010
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Remove anything within MATLAB Workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lorenz system variables
    sig = 10;
    rho = 28;
    bet = 8/3;
% Observation & Numerical variables
    var = 0.0001; % Four possible variances
    dt = 0.01; % Assignment gave dt = 0.05, but that blows up.
    x0T = [1.0, 1.0, 1.0]'; % True starting point.
    x0O = [1.1, 1.1, 1.1]'; % Erroneous observation starting point
    t_end = 5;  % Total time for trajectories
    t_obs = 2;  % Time for observation points
    f_obs = 2;  % Collect observation every f_obs time step
% Create covariance matrix as identity * variance.
    R = eye(size(x0O,1))*var;
% Define some variables to be used in plotting result
    fN = 'Minion Pro';
    fS = 12;
    mS = 2.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step A: Discretize Lorenz model using Euler forward
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% See function: lorenzO1F(x, dt, sigma, rho, beta)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step B: Compute Jacobian matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% See function: jacobien(x, dt, sigma, rho, beta)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step C: Pick initial condition (x0,y0,z0) = (1, 1, 1) and compute true
%   trajectory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_fin = t_end/dt + 1;
for k = 1:n_fin
    tT(k) = (k-1) * dt;
    if k == 1
        xT(:,k) = x0T;
    else 
        xT(:,k) = lorenzO1F(xT(:,k-1), dt, sig, rho, bet);
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step D: Generate observations with Gaussian white noise based on variance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This part generates the observations based upon the true data from
%  Step C, but adding Gaussian noise to it.
n_fin = t_obs/dt+1;
kO = 1;
for k = f_obs:f_obs:n_fin
    for i = 1:3
       v(i,1) = 0 + var.*randn(1); 
    end
    z(:,kO) = xT(:,k) + v(:);
    kO = kO + 1;
    tO(kO) = tT(k);
end
tO = tO(2:end);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step E: Forecast erroneous trajectory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This part generates an erroneous trajectory based upon the bad initial
%  conditions of (x0, y0, z0) = (1.1, 1.1, 1.1).
n_fin = t_end/dt + 1;
for k = 1:n_fin
    tE(k) = (k-1) * dt;
    if k == 1
        xE(:,k) = x0O;
    else 
        xE(:,k) = lorenzO1F(xE(:,k-1), dt, sig, rho, bet);
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step F: Asssimilate data using 4D-Var & Optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the conjugate gradient method
%alpha = 1; % Initial guess for step length
n_fin = t_obs/dt + 1;

% Set initial gradient for alpha
gradJ = grad_4DVAR(x0O,dt,sig,rho,bet, t_obs, f_obs, R, z);
p = -gradJ;
tol = 1*10^-8;
x0 = x0O;
h = eye(3);

for i = 1:300
    x0m = x0;
    
    % Compute trajectory, take points from trajectory and compute cost
    %   function
    [x, kO] = trajForCost(x0m, dt, sig,rho,bet, n_fin, f_obs);
    
    % Compute cost/objective function
    cost = costFcn(z, x, R);
    
    % Compute step length using quadratic approximation
    if i == 1
        alpha = stepGSS(gradJ, p, cost,0, i, 0, 10^-3, 100, x0m, dt, ...
            sig,rho,bet, n_fin, f_obs, z, R, tol);
    else
        alpha = stepGSS(gradJ, p, cost,costOld, i, 0, 10^-3, 100,x0m,dt, ...
            sig,rho,bet, n_fin, f_obs, z, R, tol);
    end
    
    % Compute new x0 & gradient
    x0 = x0m + alpha*p;
    gradJNew = grad_4DVAR(x0,dt,sig,rho,bet, t_obs, f_obs, R, z);
    
    if abs(norm(gradJ)/norm(gradJNew)) <= tol, break, end
    test(i) = abs(norm(x0) - norm(x0m))/norm(x0m);
    if test(i) <= 10^-3/(var/0.0001), break, end
    % Compute new p
    y = gradJNew - gradJ;
    pN = -gradJNew + (gradJNew' * gradJNew)/(gradJ'*gradJ)*p;
    
    
    gradJ = gradJNew;
    p = pN;
    costOld = cost;   
end

% Plot trajectory from analyzed x0
n_fin = t_end/dt + 1;
for k = 1:n_fin
    tA(k) = (k-1) * dt;
    if k == 1
        xA(:,k) = x0;
    else 
        xA(:,k) = lorenzO1F(xA(:,k-1), dt, sig, rho, bet);
    end
end

%Plot true trajectory
subplot(3,1,1)
    plot(tT,xT(1,:),'-k', 'LineWidth', 2, 'DisplayName', 'True Trajectory')
    hold on
        plot([2 2], [-40 40], '--r','HandleVisibility','Off')
        plot(tO, z(1,:), 'or','MarkerSize',mS,'DisplayName','Observations')
        plot(tE,xE(1,:),'--b','LineWidth',1.5,...
            'DisplayName','Erroneous Trajectory')
        plot(tA,xA(1,:),':m','LineWidth',1.5,...
            'DisplayName','Analyzed Trajectory')
    hold off
    ylim([-30 30])
    ylabel('x', 'FontWeight', 'bold')
    ax = gca;
        ax.FontSize = fS;
        ax.FontName = fN;
    titleStr = strcat('\sigma^2 =',{' '},num2str(var),...
        ', i =', {' '}, num2str(i), ',',{' '},...
        'x_0 =',{' '},'[',num2str(x0(1)),',',{' '}, ...
        num2str(x0(2)),',',{' '},num2str(x0(3)),']');
    title(titleStr)
    legend('Location','Southeast')
subplot(3,1,2)
    plot(tT,xT(2,:),'-k', 'LineWidth', 1.5)
    hold on
        plot([2 2], [-40 40], '--r')
        plot(tO, z(2,:), 'or', 'MarkerSize',mS)
        plot(tE,xE(2,:),'--b', 'LineWidth', 1.5)
        plot(tA,xA(2,:),':m', 'LineWidth', 1.5)
    hold off
    ylim([-30 30])
    ylabel('y', 'FontWeight', 'bold')
    ax = gca;
        ax.FontSize = fS;
        ax.FontName = fN;
subplot(3,1,3)
    plot(tT,xT(3,:),'-k', 'LineWidth', 1.5)
    hold on
        plot([2 2], [0 60], '--r')
        plot(tO, z(3,:), 'or', 'MarkerSize',mS)
        plot(tE,xE(3,:),'--b', 'LineWidth', 1.5)
        plot(tA,xA(3,:),':m', 'LineWidth', 1.5)
    hold off
    ylim([0 60])
    xlabel('Time, t', 'FontWeight', 'bold')
    ylabel('z', 'FontWeight', 'bold')
    ax = gca;
        ax.FontSize = fS;
        ax.FontName = fN;

%% ----------------------------- FUNCTIONS ----------------------------- %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: Euler discretization for Lorenz equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is meant to be iterated *within* a look to calculate each
% time-step of a trajectory.
% - Output is the value at next time step, x(k+1)
% - Input is the value at current time step, x(k), step size, dt, and the
%       Lorenz model parameters (sigma, rho, beta)
function xPlus = lorenzO1F(x, dt, sigma, rho, beta)
    f(1) = -sigma*(x(1) - x(2));
    f(2) = rho*x(1)  - x(2) - x(1)*x(3);
    f(3) = x(1)*x(2) - beta*x(3);
    
    xPlus(1) = x(1) + dt*f(1);
    xPlus(2) = x(2) + dt*f(2);
    xPlus(3) = x(3) + dt*f(3);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: Jacobian matrix of Lorenz equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the Jacobian (solved by hand) of the Lorenz model.
% - Output is a 3×3 matrix
% - Input is values at current time, x(k), and step size, dt, and the
%   Lorenz model parameters (sigma, rho, beta)
function DM = jacobien(x, dt, sigma, rho, beta)
    DM(1,1) = 1 - dt*sigma;
    DM(1,2) = dt*sigma;
    DM(1,3) = 0;
    DM(2,1) = dt*(rho-x(3));
    DM(2,2) = 1 - dt;
    DM(2,3) = -dt*x(1);
    DM(3,1) = dt*x(2);
    DM(3,2) = dt*x(1);
    DM(3,3) = 1 - dt*beta;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: 4D-VAR to find gradient
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function uses the 4D-VAR algorithm for nonlinear M and h to find an
% optimal gradient for minimization.
% - Output is optimal gradient, gradJ
% - Input is initial state variable, x0 (n×1), time step, dt, Lorenz model
%   parameters (sigma, rho, beta), maximum time of observations, t_obs,
%   covariance matrix, R, observations, z, 
function gradJF = grad_4DVAR(x0, dt, sig,rho,bet, t_obs, f_obs, R, z)
    n_fin = t_obs/dt + 1;
    kO = 1;
    
    for k = 1:n_fin
    % 1 & 2) Compute next step and Jacobian evaluated at next Step
        if k == 1
            x(:,k) = x0;
            DM(:,:,k) = jacobien(x0, dt, sig,rho,bet);
        else 
            x(:,k) = lorenzO1F(x(:,k-1), dt, sig,rho,bet);
            DM(:,:,k) = jacobien(x(:,k), dt, sig,rho,bet);
        end 
        
    % 3) Compute forecast error
        %  Only take at observation points!
        testObs = mod(k,f_obs);
        f(:,1) = [0; 0; 0;];
        if testObs == 0
          f(:,k) = pinv(R)*(z(:,kO) - x(:,k)); % D(H) = identity matrix
          kO = kO + 1;
        else
          f(:,k) = 0;  
        end
    end
    % 4) Solve adjoint Lagrangian system
    l(:,k) = f(:,k);
    for k = n_fin-1:-1:1
        l(:,k) = DM(:,:,k)' * l(:,k+1) + f(:,k);
    end

    % 5) Calculate gradient
    gradJ(:) = -DM(:,:,1)' * l(:,2);
    gradJF = gradJ';
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: Compute trajectory for cost computation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This computes a trajectory over n_fin steps, given an intial condition,
% using the Lorenz model.
% - Outputs are trajectory, xF, and final index value, kF
% - Inputs are number of steps, n_fin, initial condition, x0, time step,
%   dt, and the Lorenz model parameters (sigma, rho, beta)
function [xF, kF] = trajForCost(x0, dt, sigma,rho,beta, n_fin,f_obs)
    xN(:,1) = x0;
    for k = 2:n_fin
        xN(:,k) = lorenzO1F(xN(:,k-1), dt, sigma, rho, beta);
    end
    % Take points from trajectory at observation times
    x(:,1) = xN(:,1);
    kO = 2;
    for k = f_obs:f_obs:n_fin
        x(:,kO) = xN(:,k);
        kO = kO + 1;
    end
    
    xF = x;
    kF = kO;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: Compute cost function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This computes the cost/objective function for 4D-VAR
% - Output is the cost
% - Inputs are observations, z, trajectory points at the same time as the
%   observations, x, and the variance matrix, R.
function costF = costFcn(z, x, R)
% Compute cost/objective function
    cost = 0;
    for k = 1:size(z,2)
        cost = cost + (z(:,k) - x(:,k))' * pinv(R) * (z(:,k) - x(:,k));
    end
    costF = cost * 0.5;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: Golden section search for step length
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function alphaF = stepGSS(gradJ, p, cost, costOld, i, aMin, aMax,...
                    maxIter, x0, dt, sigma,rho,beta, n_fin, f_obs, z, ...
                    R, tolerance)
% This function finds optimal step length by a golden section search
% - Output is step length
% - Inputs:
% -     Gradient calculated from 4DVAR,     gradJ 
% -     Step direction,                     p
% -     Evaluation of cost function,        cost 
% -     Previous o. l. iteration's cost,    costOld
% -     Current outer loop iteration index, i
% -     Guess of minimum alpha,             aMin
% -     Guess of maximum alpha,             aMax
% -     Maximum search iterations,          maxIter
% -     Initial guess for state variable,   x0
% -     Time step size,                     dt
% -     Lorenz model parameters,            sigma, rho, & beta
% -     Number of iterations generate,      n_fin
% -     Frequency of iterations,            f_obs
% -     Observations,                       z
% -     Variance matrix,                    R
% -     Tolerance of answer,                tolerance

    % Specify golden ratio, phi
    phi = (1 + sqrt(5))/2; 

    % Initial guess of maximum alpha
    if i == 1
        b = max([0.00001*aMax, - 0.5 * cost / (gradJ' * p)]);
    else
        b = max([0.00001*aMax, 2 * (cost - costOld) / (gradJ' * p)]);
    end
    
    % Initial guess of minimum alpha
    a = 0;
    alpha = b;
    x0l = x0;

    for i = 1:maxIter
        c = b - (b - a)/phi;
        d = a + (b - a)/phi;

        % Evaluate costs of each
        xC0 = x0l + c*p;
        xC = trajForCost(xC0, dt, sigma,rho,beta, n_fin, f_obs);
        costC = costFcn(z, xC, R);
        
        xD0 = x0l + d*p;
        xD = trajForCost(xD0, dt, sigma,rho,beta, n_fin, f_obs);
        costD = costFcn(z, xD, R);
        
        % Compare costs
        if costC <= costD
            a = d;
        else
            b = c;
        end
        
        %alphaF = alpha;
        
        % Check tolerance
        if abs(c-d) <= tolerance
            break
        end
%         figure(2)
%         subplot(3,1,1)
%         plot([1:100],z(1,:),'or')
%         hold on
%             plot([1:100],xD(1,1:end-1),'--b', 'DisplayName', 'Min')
%             plot([1:100],xC(1,1:end-1),':g', 'DisplayName', 'Max')
%             legend('Location','Northeast')
%         hold off
%         subplot(3,1,2)
%         plot([1:100],z(1,:),'or')
%         hold on
%             plot([1:100],xD(2,1:end-1),'--b', 'DisplayName', 'Min')
%             plot([1:100],xC(2,1:end-1),':g', 'DisplayName', 'Max')
%             legend('Location','Northeast')
%         hold off
%         subplot(3,1,3)
%         plot([1:100],z(1,:),'or')
%         hold on
%             plot([1:100],xD(3,1:end-1),'--b', 'DisplayName', 'Min')
%             plot([1:100],xC(3,1:end-1),':g', 'DisplayName', 'Max')
%             legend('Location','Northeast')
%         hold off
        
        clear c d xC0 xC costC xD0 xD costD
    end
    
    alphaF = c;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: Quadratic approximation for step length
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function finds optimal step length by quadratic approximation
% CURRENTLY DOES NOT WORK!
function alpha = stepQuad(gradJ, p, cost, costOld, i, x0, dt, n_fin,...
                    f_obs, sigma,rho,beta, z, R, tolerance)
    % Initial guess, to be used later for alpha calculations
    if i == 1
        alphaI = - 0.5 * cost / (gradJ' * p); 
    else
        alphaI = 2 * (cost - costOld) / (gradJ' * p);
    end
    
    x0p = x0 + alphaI*p;
    [xp, kOp] = trajForCost(x0p, dt, sigma,rho,beta, n_fin, f_obs);
    costP = costFcn(z, xp, R);
    
    denominator = 2 * (costP - cost - gradJ' * p * alphaI);
    if abs(denominator) >= 1
        alpha = - ( gradJ' * p * alphaI^2 ) / ( denominator );
    else
        alpha = alphaI * 2;
    end
end