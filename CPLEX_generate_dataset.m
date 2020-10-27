tic;
fid = fopen('model_data_electricity_gas_piecewise_ignore.csv', 'w');

%% Load in power networks
bus_param=csvread('24 bus/bus_param.csv',1);
branch_param=csvread('24 bus/line_param.csv',1);
gen_param=csvread('24 bus/gen_param.csv',1);
Factor=9;
gen_param(:,2)=gen_param(:,2)/Factor;
gen_param(:,3)=gen_param(:,3)/Factor;
branch_param(:,4)=branch_param(:,4)/Factor;
No_of_buses=size(bus_param,1);
No_of_branches=size(branch_param,1);
No_of_generators=size(gen_param,1);

%% Load in parameters
No_of_nodes=20;
Total_power_load=2.2/Factor;
node_param=csvread(strcat(num2str(No_of_nodes),' node/node_param.csv'),1);
pipeline_param=csvread(strcat(num2str(No_of_nodes),' node/pipeline_param.csv'),1);
source_param=csvread(strcat(num2str(No_of_nodes),' node/source_param.csv'),1);
compressor_param=csvread(strcat(num2str(No_of_nodes),' node/compressor_param.csv'),1);

Z_list=compressor_param(:,7);
B_list=compressor_param(:,8);
alpha_list=compressor_param(:,9);
beta_list=compressor_param(:,10);
gamma_list=compressor_param(:,11);

%% Set up Big-M and count the number of facilities
Big_M=1e10;
No_of_pipelines=size(pipeline_param,1);
No_of_sources=size(source_param,1);
No_of_compressors=size(compressor_param,1);

for j=1:No_of_nodes
    fprintf(fid,'loadfactor%d,',j);
end

for j=1:No_of_buses
    fprintf(fid,'load_factor_power%d,',j);
end

fprintf(fid,'optimal_solution,');

for j=1:No_of_sources
    fprintf(fid,'source_generation%d,',j);
end

for j=1:No_of_generators-1
    fprintf(fid,'power_generation%d,',j);
end
fprintf(fid,'power_generation%d\n',No_of_generators);
%% Set up the decision_variables
nodal_pressure=sdpvar(No_of_nodes,1,'full');
source_generation=sdpvar(No_of_sources,1,'full');
pipeline_flow=sdpvar(No_of_pipelines,1,'full');
direction_flow=binvar(No_of_pipelines,1,'full');
pipeline_direction=binvar(No_of_pipelines,1,'full');
compressor_flow=sdpvar(No_of_compressors,1,'full');
compressor_H=sdpvar(No_of_compressors,1,'full');
compressor_consumption=sdpvar(No_of_compressors,1,'full');

power_generation=sdpvar(No_of_generators,1,'full');
angle=sdpvar(No_of_buses,1,'full');
branch_flow=sdpvar(No_of_branches,1,'full');
G2P_corr=[5,1,4;    % Generator Number
    3,5,11;         % Node Number
    1.111,0.638,0.895];  % Power Conversion Effiency
%% Set up upper and lower bounds for the gas nodes.
CONT1= node_param(:,4)<=nodal_pressure<=node_param(:,3);
CONT2= -pipeline_param(:,5)<=pipeline_flow<=pipeline_param(:,5);
CONT3= source_param(:,4)<=source_generation<=source_param(:,3);
CONT4= 0<=compressor_flow<=compressor_param(:,6);

%% Set up gas pressure ratio constraints at a compressor.
CONT5=[];
for i=1:No_of_compressors
    CONT5=[CONT5,(compressor_param(i,4)) * nodal_pressure(compressor_param(i,2))<= ...
        nodal_pressure(compressor_param(i,3))<=compressor_param(i,5) * nodal_pressure(compressor_param(i,2))]; %#ok<*AGROW>
end

%% Generator Output & DC power flow
CONT6= gen_param(:,3)<=power_generation<=gen_param(:,2);
CONT7=[];
for i=1:No_of_branches
    CONT7=[CONT7,branch_flow(i)==(angle(branch_param(i,1))-angle(branch_param(i,2)))/(branch_param(i,3)*10)];
end

CONT8= -branch_param(:,4)<=branch_flow<=branch_param(:,4);

%% Gas Flow Equation
CONT9=[];

for i=1:No_of_pipelines
    CONT9=[CONT9,-Big_M*direction_flow(i)<=nodal_pressure(pipeline_param(i,2))-nodal_pressure(pipeline_param(i,3)),...
        nodal_pressure(pipeline_param(i,2))-nodal_pressure(pipeline_param(i,3))<=Big_M*(1-direction_flow(i))];   % OK
    CONT9=[CONT9,-Big_M*direction_flow(i)<=pipeline_flow(i),pipeline_flow(i)<=Big_M*(1-direction_flow(i))];  %OK
end

No_of_pieces=101;

delta_p=sdpvar(No_of_pieces,No_of_nodes,'full');
phi_p=binvar(No_of_pieces,No_of_nodes,'full');
X_p=zeros(No_of_pieces,No_of_nodes);

delta_flow=sdpvar(No_of_pieces,No_of_pipelines,'full');
phi_flow=binvar(No_of_pieces,No_of_pipelines,'full');
X_flow=zeros(No_of_pieces,No_of_pipelines);

for i=1:No_of_nodes
    X_p(:,i)=linspace(node_param(i,4),node_param(i,3),No_of_pieces)';
end

for i=1:No_of_pipelines
    X_flow(:,i)=linspace(-pipeline_param(i,5),pipeline_param(i,5),No_of_pieces)';
end
hX_p=X_p.^2;
hX_flow=X_flow.^2.*sign(X_flow);

CONT9=[CONT9,delta_p(:)>=0,delta_p(:)<=1,delta_flow(:)>=0,delta_flow(:)<=1];

for i=1:No_of_pieces-1
    CONT9=[CONT9,delta_p(i+1,:)<=phi_p(i,:)];
end

for i=1:No_of_pieces
    CONT9=[CONT9,phi_p(i,:)<=delta_p(i,:)];
end

for i=1:No_of_pieces-1
    CONT9=[CONT9,delta_flow(i+1,:)<=phi_flow(i,:)];
end

for i=1:No_of_pieces
    CONT9=[CONT9,phi_flow(i,:)<=delta_flow(i,:)];
end
pipeline_flow_squared=sdpvar(No_of_pipelines,1);
for i=1:No_of_pipelines
    CONT9=[CONT9,pipeline_flow_squared(i)==hX_flow(1,i)+diff(hX_flow(:,i))'*delta_flow(1:No_of_pieces-1,i)];
    CONT9=[CONT9,pipeline_flow(i)==X_flow(1,i)+diff(X_flow(:,i))'*delta_flow(1:No_of_pieces-1,i)];
end

nodal_pressure_squared=sdpvar(No_of_nodes,1);
for i=1:No_of_nodes
    CONT9=[CONT9,nodal_pressure_squared(i)==hX_p(1,i)+diff(hX_p(:,i))'*delta_p(1:No_of_pieces-1,i)];
    CONT9=[CONT9,nodal_pressure(i)==X_p(1,i)+diff(X_p(:,i))'*delta_p(1:No_of_pieces-1,i)];
end

for i=1:No_of_pipelines
CONT9=[CONT9,pipeline_flow_squared(i)==pipeline_param(i,4)^2*(nodal_pressure_squared(pipeline_param(i,2))-nodal_pressure_squared(pipeline_param(i,3)))];
end


coeff=0.1;
rng(1);
toc;tic;
for iter=1:10000
load_factor_power=rand([No_of_buses,1])*coeff*2-coeff;
load_factor=rand([No_of_nodes,1])*coeff*2-coeff;

%% Nodal balance Equation (For Natural Gas)
node_injection=sdpvar(No_of_nodes,1,'full');
CONT10= node_injection(:)==0;

for i=1:No_of_sources
    node_injection(source_param(i,2))=node_injection(source_param(i,2))+source_generation(i);
end

for i=1:size(G2P_corr,2)
    node_param(G2P_corr(2,i),2)=0;
end

node_injection=node_injection-node_param(:,2).*(1+load_factor);

for i=1:size(G2P_corr,2)
    node_injection(G2P_corr(2,i))=node_injection(G2P_corr(2,i))-power_generation(G2P_corr(1,i))*G2P_corr(3,i)*35.3147*1e3;
end

for i=1:No_of_pipelines
    node_injection(pipeline_param(i,2))=node_injection(pipeline_param(i,2))-pipeline_flow(i);
    node_injection(pipeline_param(i,3))=node_injection(pipeline_param(i,3))+pipeline_flow(i);
end

for i=1:No_of_compressors
    node_injection(compressor_param(i,2))=node_injection(compressor_param(i,2))-compressor_flow(i)-compressor_consumption(i);
    node_injection(compressor_param(i,3))=node_injection(compressor_param(i,3))+compressor_flow(i);
end

CONT11= node_injection==0;

%% Nodal Balance Equations for power system
power_injection=sdpvar(No_of_buses,1,'full');
CONT12= power_injection==0;
for i=1:No_of_generators
    power_injection(gen_param(i,1))=power_injection(gen_param(i,1))+power_generation(i);
end

for i=1:No_of_buses
    power_injection(i)=power_injection(i)-bus_param(i,2)*(1+load_factor_power(i))*Total_power_load;
end

for i=1:No_of_branches
    power_injection(branch_param(i,1))=power_injection(branch_param(i,1))-branch_flow(i);
    power_injection(branch_param(i,2))=power_injection(branch_param(i,2))+branch_flow(i);
end
CONT13= power_injection==0;
%% Set objective and solve
objective=sum(source_param(:,5).*source_generation)+sum(gen_param(:,4).*power_generation*1e6);
Constraints=[CONT1,CONT2,CONT3,CONT4,CONT5,CONT6,CONT7,CONT8,CONT9,CONT10,CONT11,CONT12,CONT13];
ops=sdpsettings('solver','cplex','verbose',0);
epsilon=0.01;
Max_iteration=50;
consumption_old=-1*ones(No_of_compressors,1);
break_indicator=1;
for iteration=1:Max_iteration
    sol=optimize(Constraints,objective,ops); 
    if sol.problem~=0
        break_indicator=0;
        break
    end
    consumption_new=value(compressor_consumption);
    error=abs(consumption_new-consumption_old);
    if max(error)>epsilon
        for i=1:No_of_compressors
            if error(i)>epsilon
                current_flow=value(compressor_flow(i));
                current_p_out=value(nodal_pressure(compressor_param(i,3)));
                current_p_in=value(nodal_pressure(compressor_param(i,2)));
                current_H=B_list(i)*current_flow*((current_p_out/current_p_in)^(Z_list(i)/2)-1);
                current_consumption=alpha_list(i)*current_H^2+beta_list(i)*current_H+gamma_list(i);
                temp=gradient_comp_unsquared(alpha_list(i),beta_list(i),B_list(i),current_flow, ...
                                         Z_list(i),current_p_out,current_p_in);
                gradient_F=temp(1);gradient_out=temp(2);gradient_in=temp(3);
                Constraints=[Constraints,compressor_consumption(i)>= ...
                    current_consumption+gradient_F*(compressor_flow(i)-current_flow)+ ...
                    gradient_out*(nodal_pressure(compressor_param(i,3))-current_p_out)+ ...
                    gradient_in*(nodal_pressure(compressor_param(i,2))-current_p_in)];
            consumption_old=consumption_new;
            end
        end
    else
        break
    end
end
if break_indicator
for j=1:No_of_nodes
    fprintf(fid,'%d,',load_factor(j));
end

for j=1:No_of_buses
    fprintf(fid,'%d,',load_factor_power(j));
end

fprintf(fid,'%d,',value(objective));

for j=1:No_of_sources
    fprintf(fid,'%d,',value(source_generation(j)));
end

for j=1:No_of_generators-1
    fprintf(fid,'%d,',value(power_generation(j)));
end
fprintf(fid,'%d\n',value(power_generation(No_of_generators)));
end
end
toc;