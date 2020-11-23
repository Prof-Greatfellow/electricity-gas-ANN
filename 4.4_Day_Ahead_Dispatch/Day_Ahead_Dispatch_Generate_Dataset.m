fid = fopen('training_set.csv', 'w');

%% 初始化参数
for i=1:34
    eval(strcat('CONT',num2str(i),'=[];')); % 约束变量集
end

All_data=xlsread('Power_Gas_Data.xlsx','A4:R169');


%% 导入电网参数
Data_of_traditional_generators=All_data(1:8,6:16);  % 传统发电机（燃气、燃煤）的所有参数 
Data_of_branches=All_data(1:38,1:4);            % 电网支路的所有参数
Data_of_buses=All_data(44:67,15:17);            % 电网节点的所有参数
Data_of_time_dependence=All_data(16:39,6:9);    % 与时间相关的电网、气网参数（负荷为主）
No_of_buses=size(Data_of_buses,1);              % 节点数量
No_of_traditional_generators=size(Data_of_traditional_generators,1);    % 传统发电机（燃气、燃煤）的数量 
No_of_branches=size(Data_of_branches,1);        % 电网节点的数量
Power_base=0.1;                                 % 电网有功功率标么值 (单位：GW)
t=Data_of_time_dependence(:,1);                 % 时间戳

for j=1:length(t)
    fprintf(fid,'powerfactor%d,',j);
end

for j=1:length(t)
    fprintf(fid,'gasfactor%d,',j);
end

for j=1:No_of_traditional_generators
    for t0=1:length(t)
        fprintf(fid,'power_generation%d_%d,',j,t0);
    end
end

fprintf(fid,'optimal_solution\n');

%% 导入气网参数
Data_of_pipelines=All_data(15:38,11:18);        % 气网管道的所有参数
Data_of_compressors=All_data(44:46,1:3);        % 气网压缩机的所有参数
Data_of_storages=All_data(57:60,1:7);           % 气网储能的所有参数
Data_of_wells=All_data(51:52,1:4);              % 气网气井的所有参数
Data_of_gas_nodes=All_data(44:63,9:13);         % 气网节点的所有参数
No_of_wells=size(Data_of_wells,1);              % 气井数量
No_of_storages=size(Data_of_storages,1);        % 气网储能数量
No_of_gas_nodes=size(Data_of_gas_nodes,1);      % 气网节点数量
No_of_pipelines=size(Data_of_pipelines,1);      % 气网管道的数量
No_of_compressors=size(Data_of_compressors,1);  % 压缩机的数量

Wind_turbine_located=All_data(9:10,6);
No_of_wind_turbines=size(Wind_turbine_located,1); % 确定风机数量

%% 导入风机参数（请在最后的多程序运算中删除）
WindFarmOutput=ones(1,24)*0.5;

%% 设置目标函数
CuP=Data_of_traditional_generators(:,4);              % 每个机组出力的边际成本 (M$/GWh)
CkA=Data_of_buses(:,3);                   % 电网切负荷成本 (M$/GWh)
CwG=Data_of_wells(:,4);                   % 天然气切负荷成本 (M$/MSm3)
CsS=Data_of_storages(:,7);                % 天然气储能成本 (M$/MSm3)
CiB=Data_of_gas_nodes(:,5);               % 电网切负荷成本 (M$/MSm3)

Pmax=Data_of_traditional_generators(:,2);            % 机组出力上限 (GW)
Pmin=Data_of_traditional_generators(:,3);            % 机组出力下限 (GW)

c=binvar(No_of_traditional_generators,length(t),'full');       % 机组是否正在运行的整数变量（整个模型中唯一与场景无关的变量）
pp=sdpvar(No_of_traditional_generators,length(t),'full');    % 机组出力
pg=sdpvar(No_of_wells,length(t),'full');         % 气井产气量 (MSm3/h)
np=sdpvar(No_of_buses,length(t),'full');         % 电网切负荷 (GWh)
qst_out=sdpvar(No_of_storages,length(t),'full'); % 储能向管道的输气量
ng=sdpvar(No_of_gas_nodes,length(t),'full');     % 气网切负荷 (MSm3)
wc=sdpvar(No_of_wind_turbines,length(t),'full'); % 弃风量 (GW)
Cwu=[0;0];          % 弃风成本，暂时这样定义
CONT1=[CONT1,pp(:)>=0,ng(:)>=0,np(:)>=0,wc(:)>=0];

Objective=sum(CuP'*diag(Pmin)*c);       % 目标函数中与场景无关的成本
Objective=Objective+sum(CuP'*pp+CwG'*pg+CkA'*np+CsS'*qst_out+CiB'*ng+Cwu'*wc); % 目标函数中与场景有关的成本

%% 能量约束（第1部分）: 机组出力约束
rup=sdpvar(No_of_traditional_generators,length(t),'full');     % 上爬坡率 (GW)
rdown=sdpvar(No_of_traditional_generators,length(t),'full');   % 下爬坡率 (GW)
CONT2=[CONT2,rup(:)>=0,rdown(:)>=0];

Eq=pp(:,:)-diag(Pmax-Pmin)*c;      % 即：pp<=c(Pmax-Pmin)。限制机组出力
CONT3=[CONT3,Eq(:)<=0];    
pp_initial=sdpvar(8,1,'full');
CONT3=[CONT3,Pmin<=pp_initial,pp_initial<=Pmax];
for t0=2:length(t)
    CONT4=[CONT4,pp(:,t0)+c(:,t0).*Pmin==pp(:,t0-1)+c(:,t0-1).*Pmin+rup(:,t0)-rdown(:,t0)];      % 出力与爬坡的关系
end              %#ok<*AGROW> % Constraints as in Eq.(2)
CONT4=[CONT4,pp(:,1)+c(:,1).*Pmin==pp_initial+rup(:,1)-rdown(:,1)];    %出力与爬坡的关系（在t=1时刻）

RU=Data_of_traditional_generators(:,8);                % 上爬坡上限 (GWh)
RD=Data_of_traditional_generators(:,7);                % 下爬坡上限 (GWh)

for t0=1:length(t)
	CONT5=[CONT5,rup(:,t0)<=RU];       % 限制上爬坡量
	CONT6=[CONT6,rdown(:,t0)<=RD];     % 限制下爬坡量
end


Wdown=Data_of_wells(:,2);          % 气井出力下限 (MSm3/h)
Wup=Data_of_wells(:,3);            % 气井出力上限 (MSm3/h)

for t0=1:length(t)
    CONT7=[CONT7,Wdown<=pg(:,t0)<=Wup];   % 限制气井出力大小
end


%% 能量约束（第2部分）: 气网能量约束
sl=sdpvar(No_of_storages,length(t),'full');    % 储能储量 (MSm3)
qst_in=sdpvar(No_of_storages,length(t),'full');   % 注入燃气量 (MSm3/h)
CONT8=[CONT8,sl(:)>=0];
Ssdown=Data_of_storages(:,2);      % 储能储量下限 (MSm3)
Ssup=Data_of_storages(:,3);        % 储能储量上限 (MSm3)
IR=Data_of_storages(:,4);      % 最大注入燃气量 (MSm3/h)
WR=Data_of_storages(:,5);      % 最大抽取燃气量 (MSm3/h)

for t0=2:length(t)
    CONT9=[CONT9,sl(:,t0)==sl(:,t0-1)+qst_in(:,t0)-qst_out(:,t0),Ssdown<=sl(:,t0)<=Ssup];   % 储能储量平衡方程
    CONT10=[CONT10,0<=qst_in(:,t0)<=IR,0<=qst_out(:,t0)<=WR]; % 储能进出约束
end

sl_initial=Data_of_storages(:,6);   % 储能初值

CONT11=[CONT11,sl(:,1)==sl_initial+qst_in(:,1)-qst_out(:,1),Ssdown<=sl(:,1)<=Ssup];
CONT12=[CONT12,0<=qst_in(:,1)<=IR,0<=qst_out(:,1)<=WR];

%The above constraints correspond to Eq.(8)(9)

R=(8.31446e-5)/(16.0425e-3);        % 理想气体常数 (m3bar/kgK)
T=273.15+8;                           % 温度 (K)
Z=0.8;                                % 压缩系数 (理想情况下Z=1)
rou0=0.7156;                         % 理想情况下的气体密度 (kg/m3)
qij_in=sdpvar(No_of_pipelines,length(t),'full');    % 管道ij在节点i的进气量 (MSm3/h)
qij_out=sdpvar(No_of_pipelines,length(t),'full');   % 管道ij在节点j的出气量 (MSm3/h)
qij_average=sdpvar(No_of_pipelines,length(t),'full');
qij_average(:)=(qij_in(:)+qij_out(:))/2;
p=sdpvar(No_of_gas_nodes,length(t),'full');         % 节点气压(bar)
pij=sdpvar(No_of_pipelines,length(t),'full');       % 管道平均气压 (bar)
mij=sdpvar(No_of_pipelines,length(t),'full');       % 管道平均燃气质量 (MSm3)

for j=1:No_of_pipelines
    pij(j,:)=(p(Data_of_pipelines(j,1),:)+p(Data_of_pipelines(j,2),:))/2;  %管道平均气压表达式
    if ~isnan(Data_of_pipelines(j,3))       %如果管道不在压缩机上
        mij(j,:)=(pi/4*Data_of_pipelines(j,3)*Data_of_pipelines(j,4)^2/R/T/Z/rou0)*pij(j,:)/1e6;
        % 质量与气压的关系（简单而言就是pV=nRT)
    end
end

for t0=2:length(t)
    for ll=1:No_of_pipelines
        if ~isnan(Data_of_pipelines(ll,3))
            CONT17=[CONT17,mij(ll,t0,:)==mij(ll,t0-1,:)-qij_in(ll,t0,:)+qij_out(ll,t0,:)];
            % 管道气压与进出燃气的关系式
        end  
    end
end

mij_initial=sdpvar(No_of_pipelines,1,'full');
p_initial=sdpvar(No_of_gas_nodes,1,'full');
CONT13=[CONT13,Data_of_gas_nodes(:,3)<=p_initial<=Data_of_gas_nodes(:,2)];
CONT14=[CONT14,mij_initial>=0]; %初始质量

p_initial=[60.0469;60.0173;60.0042;53.1064;20.0122;30.0085;45.0116;53.1017;
   45.0000;66.1992;66.1984;66.1974;66.1950;53.0914;52.9544;52.5592;66.1365;
   76.4437;69.2551;65.4440];
for j=1:No_of_pipelines
     if ~isnan(Data_of_pipelines(j,3))
         mij_initial(j,1)=(pi/4*Data_of_pipelines(j,3)*Data_of_pipelines(j,4)^2 ...
         /R/T/Z/rou0)*(p_initial(Data_of_pipelines(j,1))+p_initial(Data_of_pipelines(j,2)))/2/1e6;
            CONT15=[CONT15,mij(j,1)==mij_initial(j,1)-qij_in(j,1)+qij_out(j,1)];
     end
end

%% 假设节点气流方向不随着时间、场景而改变
flow_direction=Data_of_pipelines(:,7);
for j=1:No_of_pipelines
    CONT16=[CONT16,flow_direction(j)*p_initial(Data_of_pipelines(j,1))>=flow_direction(j)*p_initial(Data_of_pipelines(j,2))];
    CONT16=[CONT16,flow_direction(j)*p(Data_of_pipelines(j,1),:)>=flow_direction(j)*p(Data_of_pipelines(j,2),:)];
    CONT16=[CONT16,flow_direction(j)*qij_average(j,:)>=0];
end

%% 网络拓扑约束（第2部分）：能流
Fkl=Data_of_branches(:,4);
theta=sdpvar(No_of_buses,length(t),'full');
fp=sdpvar(No_of_branches,length(t),'full'); %电网潮流
for t0=1:length(t)
    CONT20=[CONT20,-Fkl<=fp(:,t0)<=Fkl];
end

for j=1:No_of_branches
    CONT21=[CONT21,fp(j,:)==(theta(Data_of_branches(j,1),:)-theta(Data_of_branches(j,2),:))/Data_of_branches(j,3)];
end

%% 网络拓扑约束（第3部分）：节点约束
for t0=1:length(t)
    CONT22=[CONT22,Data_of_gas_nodes(:,3)<=p(:,t0)<=Data_of_gas_nodes(:,2)];
end

for j=1:No_of_compressors
    CONT23=[CONT23,p(Data_of_compressors(j,1),:)<=p(Data_of_compressors(j,2),:)<=p(Data_of_compressors(j,1),:)*Data_of_compressors(j,3)];
end
penalty_factor=[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]'*ones(1,24)*2e-6;

for j=1:No_of_pipelines
    if flow_direction(j)==1
        CONST=sqrt((3600/1e6)^2*1e5*(pi/4)^2*Data_of_pipelines(j,4)^5/Data_of_pipelines(j,3)/Data_of_pipelines(j,5)/R/T/Z/rou0^2);
         if CONST<sqrt(1e-13)
             display('error!');
         end
        for t0=1:length(t)
            CONT34=[CONT34,norm([1/CONST*qij_average(j,t0),p(Data_of_pipelines(j,2),t0)],2)...
                <=p(Data_of_pipelines(j,1),t0)]; 
            CONT34=[CONT34,1/CONST*qij_average(j,t0)+p(Data_of_pipelines(j,2),t0)>=p(Data_of_pipelines(j,1),t0)];
            Objective=Objective+penalty_factor(j,t0)*(p(Data_of_pipelines(j,1),t0)-p(Data_of_pipelines(j,2),t0));
        end
    elseif flow_direction(j)==-1
        CONST=sqrt((3600/1e6)^2*1e5*(pi/4)^2*Data_of_pipelines(j,4)^5/Data_of_pipelines(j,3)/Data_of_pipelines(j,5)/R/T/Z/rou0^2);
         if CONST<sqrt(1e-13)
             display('error!');
         end
          for t0=1:length(t)
              CONT34=[CONT34,1/CONST*qij_average(j,t0)+p(Data_of_pipelines(j,2),t0)<=p(Data_of_pipelines(j,1),t0)];
              CONT34=[CONT34,norm([1/CONST*qij_average(j,t0),p(Data_of_pipelines(j,1),t0)],2)<=p(Data_of_pipelines(j,2),t0)];
          Objective=Objective+penalty_factor(j,t0)*(p(Data_of_pipelines(j,2),t0)-p(Data_of_pipelines(j,1),t0));
          end
    end
end

sum_mij=mij(1,24);
sum_mij_initial=mij_initial(1,1);

for j=1:No_of_pipelines
    if ~isnan(Data_of_pipelines(j,3))
        sum_mij=sum_mij+mij(j,24);
        sum_mij_initial=sum_mij_initial+mij_initial(j);
    end
end
sum_mij=sum_mij-mij(1,24);
sum_mij_initial=sum_mij_initial-mij_initial(1,1);
CONT34=[CONT34,sum_mij>=sum_mij_initial];
                                          
%% 网络拓扑约束（第1部分）：节点平衡方程
rng(11);
No_of_total_samples=10000;
for iter=1:No_of_total_samples
CONT_IN_ITER=[];
coeff=0.05;
power_random=repmat(rand([1,length(t)])*coeff*2-coeff,No_of_buses,1);
gas_random=repmat(rand([1,length(t)])*coeff*2-coeff,No_of_gas_nodes,1);
%coeff_random=[-0.014543177,-0.001505967,-0.095728368,0.054179327,-0.010676401,-0.05232802,0.004573537,0.068010039,-0.080667087,-0.016786791,-0.017169382,0.01117177,0.037443677,0.010908408,0.071978547,0.082306737,-0.077455883,-0.060086125,0.025047892,-0.071135621,-0.007947131,0.094142994,0.057843909,-0.007643817,0.073299105,0.017070787,-0.077443272,0.005375228,0.039849971,0.098027011,0.063618332,0.049285089,-0.030742331,-0.007935341,0.004357366,-0.058442356,-0.021675493,-0.024309012,0.006386945,-0.026773437,0.056916603,0.05682926,0.091892814,0.010806745,0.016219822,0.056655795,-0.092633379,0.095361228];
%coeff_random=ones([1,length(t)*2])*-0.03;
%power_random=repmat(coeff_random(1:24),No_of_buses,1);
%gas_random=repmat(coeff_random(25:48),No_of_gas_nodes,1);
LkP=Data_of_buses(:,2)*Data_of_time_dependence(:,3)'.*(1+power_random);       % 电网负荷需求
LiG=Data_of_gas_nodes(:,4)*Data_of_time_dependence(:,2)'.*(1+gas_random);    % 气网负荷需求
no_meaning=sdpvar(No_of_buses,length(t),'full');


CONT_IN_ITER=[CONT_IN_ITER,no_meaning(:)==0];
temp=no_meaning;        %初始化temp变量，指的是进入该节点的电能

for j=1:No_of_branches
    %添加潮流
    temp(Data_of_branches(j,1),:)=temp(Data_of_branches(j,1),:)-fp(j,:);
    temp(Data_of_branches(j,2),:)=temp(Data_of_branches(j,2),:)+fp(j,:);
end

for j=1:No_of_traditional_generators
    %添加机组出力
    temp(Data_of_traditional_generators(j,1),:)=temp(Data_of_traditional_generators(j,1),:)+pp(j,:)+c(j,:)*Pmin(j);
end

for j=1:No_of_buses
    temp(j,:)=temp(j,:)+np(j,:);
end

for j=1:No_of_wind_turbines
    temp(Wind_turbine_located(j),:)=temp(Wind_turbine_located(j),:)+WindFarmOutput-wc(j,:);
end

CONT_IN_ITER=[CONT_IN_ITER,temp==LkP];

% 对于Compressor部分，暂时认为不需要消耗能量，同时仅满足约束flow in=flow out。
for j=1:No_of_pipelines
    if isnan(Data_of_pipelines(j,3))
        CONT_IN_ITER=[CONT_IN_ITER,qij_out(j,:)==qij_in(j,:)];
    end
end

no_meaning2=sdpvar(No_of_gas_nodes,length(t),'full');
CONT_IN_ITER=[CONT_IN_ITER,no_meaning2(:)==0];
temp2=no_meaning2;

for j=1:No_of_pipelines
    temp2(Data_of_pipelines(j,1),:)=temp2(Data_of_pipelines(j,1),:)-qij_out(j,:);
    temp2(Data_of_pipelines(j,2),:)=temp2(Data_of_pipelines(j,2),:)+qij_in(j,:);
end

for j=1:No_of_storages
    temp2(Data_of_storages(j,1),:)=temp2(Data_of_storages(j,1),:)+qst_out(j,:)-qst_in(j,:);
end

for j=1:No_of_wells
    temp2(Data_of_wells(j,1),:)=temp2(Data_of_wells(j,1),:)+pg(j,:);
end

for j=1:No_of_gas_nodes
    temp2(j,:)=temp2(j,:)+ng(j,:);
end

phi=[Data_of_traditional_generators(5,10),Data_of_traditional_generators(1,10),Data_of_traditional_generators(4,10)];
    Correlation=[2,5,14;                % 第一行：对应的气网节点
             5,1,4;                 % 第二行：对应的电网节点
            15,1,13];               % 第三行：对应的机组序号
for j=1:size(Correlation,2)
    temp2(Correlation(1,j),:)=temp2(Correlation(1,j),:)-phi(1,j)*(pp(Correlation(2,j),:)+c(Correlation(2,j),:)*Pmin(Correlation(2,j)));
end

CONT_IN_ITER=[CONT_IN_ITER,temp2==LiG];

%% 设置非线性约束（具体计算公式请参加文章）
                                                                                       
Constraints=[CONT1,CONT2,CONT3,CONT4,CONT5,CONT6,CONT7,CONT8,CONT9,CONT10,CONT11,CONT12,CONT13,CONT14,...
    CONT15,CONT16,CONT17,CONT18,CONT19,CONT20,CONT21,CONT22,CONT23,CONT24,CONT25,CONT26,CONT27,CONT28,...
    CONT29,CONT30,CONT31,CONT32,CONT33,CONT34,CONT_IN_ITER];

%% 设置气网流量的初末关系式
ops=sdpsettings('solver','cplex','verbose',0);
sol=optimize(Constraints,Objective,ops);

%% 求解器计算
if sol.problem==0
    Objective_value=value(Objective);
    Max_violations=[];
    for j=1:No_of_pipelines
        if flow_direction(j)==1
            CONST=sqrt((3600/1e6)^2*1e5*(pi/4)^2*Data_of_pipelines(j,4)^5/Data_of_pipelines(j,3)/Data_of_pipelines(j,5)/R/T/Z/rou0^2);        
            for t0=1:length(t)
                Objective_value=Objective_value-penalty_factor(j,t0)*(value(p(Data_of_pipelines(j,1),t0)-p(Data_of_pipelines(j,2),t0)));
                Max_violations=[Max_violations,(-(1/CONST*value(qij_average(j,t0)))^2-value(p(Data_of_pipelines(j,2),t0))^2+value(p(Data_of_pipelines(j,1),t0))^2)/value(p(Data_of_pipelines(j,1),t0))^2];
            end
        elseif flow_direction(j)==-1
            CONST=sqrt((3600/1e6)^2*1e5*(pi/4)^2*Data_of_pipelines(j,4)^5 ...
                 /Data_of_pipelines(j,3)/Data_of_pipelines(j,5)/R/T/Z/rou0^2);
            for t0=1:length(t)
                Objective_value=Objective_value-penalty_factor(j,t0)*(value(p(Data_of_pipelines(j,2),t0)-p(Data_of_pipelines(j,1),t0)));
                Max_violations=[Max_violations,(-(1/CONST*value(qij_average(j,t0)))^2-value(p(Data_of_pipelines(j,1),t0))^2+value(p(Data_of_pipelines(j,2),t0))^2)/value(p(Data_of_pipelines(j,2),t0))^2];
            end
        end
    end
    for i=1:length(t)
        fprintf(fid,'%d,',power_random(1,i));
    end
    for i=1:length(t)
        fprintf(fid,'%d,',gas_random(1,i));
    end

    for jj=1:No_of_traditional_generators
        for i=1:length(t)
            fprintf(fid,'%d,',value(pp(jj,i))+value(c(jj,i))*Pmin(jj));
        end
    end

    fprintf(fid,'%d\n',Objective_value);
else
    fprintf('error!');
end

end