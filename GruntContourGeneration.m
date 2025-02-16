%% THRUST CHAMBER SIZING PROCEDURE

% Aim: defining thrust chamber geometry from F, O/F ratio, p_0, r_cc

% Intervention needed in STEP 1 (CEA) and in STEP 3 (Getting angles from
% table in chapter 8 notes)

format long; clear all; close all; clc;

%% CONSTANTS
P_a      = 101325;      % Pa

%% INPUT
F        = 5000;        % N 
OF_ratio = 2.25; 
p_0      = 25e5;      % Pa 
p_e      = P_a;
R_c      = 0.055;        % m (comb. chamb. radius)


% STATES
% 0  Combustion chamber after combustion
% T  Throat
% E  Exit

%% STEP 1: CEA (MASS FLOW INDEPENDENT)

% Go to https://cearun.grc.nasa.gov and run a simulation with the data
% above.
% 
% With the following input:
% P_0                                            400 psia
% o/f                                            2.25
% Exit conditions: P_c/P_e                       27.239082161361953
% Consider Ionized Species as possible products? Yes
% 
% we get the following results:

T_0    = 3402.83;     % K
gamma  = 1.2230;      %                 (combustion chamber)
Cp     = 2.0909e3;    % J/(kg)(K)       (combustion chamber)

v_e    = 2505.8;      % m/s             (exit isp)
M_e    = 2.672;
eps    = 4.0337;      % expansion ratio (exit Ae/A_t)

%% STEP 2: MASS FLOW DEPENDENT CALCULATIONS

% First get R
Cv = Cp/gamma;
R = Cp-Cv;             % Specific gas constant

% We can calculate the needed mass flow for the required thrust
m_dot = F/v_e;

% That conditions throat area, and we can obtain it from the chocked flow
% mass flow rate equation.
% m_dot = p_0*A_t/sqrt(T_0)*sqrt(gamma/R*(2/(gamma+1))^((gamma+1)/(gamma-1)));

A_t =m_dot/(p_0/sqrt(T_0)*sqrt(gamma/R*(2/(gamma+1))^((gamma+1)/(gamma-1))));

% The compression ratio will be
eps_c = (pi*R_c^2)/A_t;

% Combustion chamber sizing
% Needed combustion chamber volume can be calculated in the following way
% V_chamber = L_characteristic * A_throat
% For LOX/RP1 the characteristic length is 76-102 cm. Lets pick 90cm.
L_characteristic = 1;             % m (for RP1-LOX should be 0.9m, but...
                                  %   being conservative, let's go with 1.)
V_chamber = L_characteristic*A_t; % m^3

% If we want the combustion chamber radius to be r_cc, then we need a
% length:
L_c = V_chamber/(pi*R_c^2);

%% STEP 3: GEOMETRICAL PARAMETERS FOR THE NOZZLE

R_t = sqrt(A_t/pi);
R_e = sqrt(eps)*R_t;

% Arbitrary
Theta_c = 40;    % deg
% FROM THE TABLE IN CHAPTER 8 NOTES
Theta_e = 15;     % deg

%% STEP 4: Solving geometry

syms x A1 A2 B1 B2 C1 C2 D1 D2 xa xab xcd xd;
ya     = A1*x + A2;
dya_dx = diff(ya,x);
yb     = -sqrt((1*R_t)^2-(x-B1)^2) + B2;
dyb_dx = diff(yb,x);
yc     = -sqrt((1*R_t)^2-(x-C1)^2) + C2;
dyc_dx = diff(yc,x);
yd     = D1*x+D2;
dyd_dx = diff(yd,x);

% Solving throat
S = solve([subs(yb, x, 0)      == R_t,...
           subs(yc, x, 0)      == R_t,...
           subs(dyb_dx, x, 0)  == 0,...
           subs(dyc_dx, x, 0)  == 0],...
           [B1 B2 C1 C2]);
yb     = subs(yb,[B1,B2],[S.B1,S.B2]);
dyb_dx = diff(yb,x);
yc     = subs(yc,[C1,C2],[S.C1,S.C2]);
dyc_dx = diff(yc,x);

% Solving throat circunferences' limits
S = vpasolve([subs(dyb_dx, x, xab)  == -tand(Theta_c),...
           subs(dyc_dx, x, xcd)  == tand(Theta_e)],...
           [xab xcd]);
clear xab xcd
xab = S.xab; xcd = S.xcd;
clear S

% Solving diverging section

S = solve([subs(dyd_dx, x, xd)  == tand(Theta_e), ...
           subs(yd, x, xcd)  == subs(yc,x, xcd),...
           subs(yd, x, xd)  == R_e],...
           [D1 D2 xd]);

yd     = subs(yd,[D1,D2],[S.D1,S.D2]);
dyd_dx = diff(yd,x);

clear xd
xd = S.xd;
clear S

% Solving converging section
S = solve([subs(dya_dx, x, xab) == -tand(Theta_c),...
           subs(ya, x, xab)  == subs(yb,x, xab),...
           subs(ya, x, xa)  == R_c],...
           [A1 A2 xa]);

ya     = subs(ya,[A1,A2,xa],[S.A1,S.A2,S.xa]);
dya_dx = diff(ya,x);

clear xa
xa = S.xa;
clear S


%% OUTPUT

xcc = xa-L_c;
xx=linspace(xa-L_c,xd);
nn=nozzle(xx,ya,yb,yc,yd,xa,xab,xcd,xd,L_c,R_c);
plot([fliplr(xx),xx], [fliplr(-nn), nn], 'linewidth', 2)
hold on;
xlabel('[m]')
ylabel('[m]')

xx_numeric = double(xx)*39.3701;
nn_numeric = double(nn)*39.3701;
writematrix([xx_numeric; nn_numeric]', 'output.csv');

axis equal
ax=gca;
xlim (1.2*ax.XLim)
ylim([-1.2*max([R_c,R_e]),1.2*max([R_c,R_e])])

disp('THRUST CHAMBER SIZING');
disp(' ');
disp(['For a RP-1/LOX engine with ' num2str(F/1000) ' kN of thrust, ' num2str(OF_ratio) ' oxidizer to fuel ratio,']);
disp(['with a chamber pressure of ' num2str(p_0/1e5) ' bars, if the combustion chamber is ']);
disp([num2str(2*R_c*1000) ' mm in diameter, the geometrical parameters of the thrust chamber are']);
disp('the following:');
disp(' ');
disp(' ');
disp(['Total length                         ' num2str(double(-xa+L_c+xd)) ' m'])
disp('');
disp(['Combustion chamber length            ' num2str(L_c) ' m'])
disp(' ')
disp(['Convergent section''s length          ' num2str(double(-xa)) ' m'])
disp(['Compression ratio                    ' num2str(eps_c)])
disp(['Convergent section''s angle (θ_c)     ' num2str(Theta_c) ' deg'])
disp(' ')
disp(['Throat radius                        ' num2str(R_t*1000) ' mm'])
disp(' ')
disp(['Expansion section''s length (L_n)     ' num2str(double(xd)) ' m'])
disp(['Expansion ratio                      ' num2str(eps)])
disp(['θ_e                                  ' num2str(Theta_e) ' deg'])
disp(['Exit radius                          ' num2str(R_e*1000) ' mm'])
disp(' ')
disp(' ')
disp(' ')

disp('Other useful parameters:')
disp(' ')
disp(['Mass flow rate                       ' num2str(m_dot) ' kg/s'])
disp(['Throat area                          ' num2str(A_t) ' m^2'])

%% FUNCTION DECLARATION

function yy = nozzle(xx,ya,yb,yc,yd,xa,xab,xcd,xd,L_c,R_c)
    yy=zeros(size(xx));
    for i = 1:length(xx)
        if xx(i)>=xa-L_c && xx(i)<xa
            yy(i) = R_c;
        elseif xx(i)>=xa && xx(i)<xab
            yy(i) = subs(ya,xx(i));
        elseif xx(i)>=xab && xx(i)<0
            yy(i) = subs(yb,xx(i));
        elseif xx(i)>=0 && xx(i)<xcd
            yy(i) = subs(yc,xx(i));
        elseif xx(i)>=xcd && xx(i)<=xd
            yy(i) = subs(yd,xx(i));
        end
    end
end


%% OUTPUT WITH DEFINITIVE DATA
% 
% THRUST CHAMBER SIZING
%  
% For a RP-1/LOX engine with 7.5 kN of thrust, 2.25 oxidizer to fuel ratio,
% with a chamber pressure of 27.6 bars, if the combustion chamber is 
% 120 mm in diameter, the geometrical parameters of the thrust chamber are
% the following:
%  
%  
% Total length                         0.37268 m
% Combustion chamber length            0.16069 m
%  
% Convergent section's length          0.071931 m
% Compression ratio                    6.2232
% Convergent section's angle (θ_c)     30 deg
%  
% Throat radius                        24.0517 mm
%  
% Expansion section's length (L_n)     0.14006 m
% Expansion ratio                      4.7409
% θ_n                                  20 deg
% θ_e                                  8 deg
% Exit radius                          52.3692 mm
%  
%  
%  
% Other useful parameters:
%  
% Mass flow rate                       2.8335 kg/s
% Throat area                          0.0018174 m^2